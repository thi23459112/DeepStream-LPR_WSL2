"""
================================================================================
 probes.py — DeepStream Pad Probe 集中區（追蹤、車牌組字、截圖、OSD）
================================================================================
本模組掛在 pipeline 各推論元件的 src/sink pad 上，負責「純 Python 端」的後處理：

  1. boxmot_pgie_src_probe / tracker_src_pad_buffer_probe
       PGIE（車輛偵測）之後做多物件追蹤（BoxMOT 或 nvtracker），
       並把每台車的 ROI 命中、方向、車種票數累積到 track_history。
  2. expand_plate_probe
       SGIE 車牌偵測之後，把車牌框往外擴 10%，避免字元貼邊被裁掉。
  3. assemble_plate_probe
       SGIE 字元偵測之後，把字元組成車牌字串、配對到車輛，並做車牌/車種截圖。
  4. per_cam_osd_probe
       demux 後每一路畫面上疊加 FPS 文字。

截圖機制（重點）：
  畫面影像「不是」從 batch buffer 直接讀，而是由 main.py 在 streammux 之前
  用 tee 分出一條 appsink 分支，把每格 RGBA（系統記憶體）連同它的 PTS 存進
  g_frame_buffer。本模組各 probe 再以 frame_meta.buf_pts 去找「PTS 最接近的那格」。
  以 PTS（時間戳）對齊，而非用計數器；如此即使 appsink 因 leaky 丟幀、或啟動時
  差一格，也只會「少存幾格」而不會「對到錯誤的畫面」——這對車牌小框的裁切尤其重要。
================================================================================
"""
import os
import time
import threading
import cv2
import numpy as np
from collections import Counter, deque
from gi.repository import Gst
import pyds

from logic.color import get_class_color, CLASS_MAP, NUM_MAP
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, _finalize_one, flush_pending_to_db,
    track_history, pending_records, last_flush_times, fps_streams, local_id_maps
)

# ---- 全域狀態 ----
g_last_fps_print_time = time.time()   # 上次印出 FPS 報告的時間（每 30 秒印一次）

# ---- 推論元件的 unique-component-id（對應各 nvinfer 的 gie-unique-id）----
_UID_VEHICLE = 1   # PGIE：車輛
_UID_PLATE   = 2   # SGIE：車牌框
_UID_CHAR    = 3   # SGIE：車牌字元

# ---- 截圖參數 ----
_JPEG_QUALITY = 70             # 存檔 JPEG 品質
_PLATE_BOTTOM_PAD_RATIO = 0.30 # 車種截圖在車牌下緣額外往下補的比例（多帶一點車尾/車頭）

# =====================================================================
# ⭐ 影格緩存（依「PTS / 時間戳」對齊，而非脆弱的計數器）
#    appsink（streammux 前）把每格 RGBA 連同它的 buf.pts 存進來；
#    probe 端用 frame_meta.buf_pts 找「最接近的那格」。
#    同一格畫面在 appsink 與 probe 兩端的 PTS 是同一個值，
#    所以即使 appsink 因 leaky 丟了幀、或啟動時差一格，也不會對錯。
# =====================================================================
_FRAME_BUFFER_PER_CAM = 64          # 每路保留最近的幀數（環形緩存上限）
_PTS_TOL_NS = 40_000_000            # 容許的 PTS 誤差（ns）≈ 40ms（<1 格@15fps≈66ms）

g_frame_buffer = {}                 # pad_index -> deque([(pts, frame_rgba), ...])
g_frame_lock   = threading.Lock()   # 保護 g_frame_buffer（appsink 執行緒寫、probe 執行緒讀）


def store_frame_for_pad(pad_index, pts, frame):
    """
    appsink 回調呼叫：把一格 RGBA 連同其 PTS 存入該路的環形緩存。

    參數：
        pad_index (int): 哪一路 cam
        pts (int):       該幀的 PTS（GStreamer buffer.pts，單位 ns）
        frame (ndarray): 該幀的 RGBA 影像（H, W, 4）
    """
    if pts is None or pts == Gst.CLOCK_TIME_NONE:
        return
    with g_frame_lock:
        dq = g_frame_buffer.get(pad_index)
        if dq is None:
            # 第一次見到這路：建立固定長度的環形緩存（自動丟最舊的）
            dq = deque(maxlen=_FRAME_BUFFER_PER_CAM)
            g_frame_buffer[pad_index] = dq
        dq.append((int(pts), frame))


def _lookup_frame_by_pts(pad_index, target_pts):
    """
    依 PTS 找該路「最接近的那格」畫面。

    參數：
        pad_index (int):   哪一路 cam
        target_pts (int):  目標 PTS（通常為 frame_meta.buf_pts）

    返回：
        ndarray | None：找到夠接近（誤差 ≤ _PTS_TOL_NS）的畫面則回傳，
                        否則回 None（寧可不截，也不要裁到錯誤的畫面）。
    """
    if target_pts is None or target_pts == Gst.CLOCK_TIME_NONE:
        return None
    dq = g_frame_buffer.get(pad_index)
    if not dq:
        return None

    target = int(target_pts)
    best = None
    best_delta = None
    # 線性掃描環形緩存，挑出 PTS 距離 target 最小的那格
    with g_frame_lock:
        for pts, frame in dq:
            d = pts - target
            if d < 0:
                d = -d
            if best_delta is None or d < best_delta:
                best_delta = d
                best = frame

    # 超過容許誤差代表「對應的那格已被丟掉」→ 回 None，不裁錯
    if best is None or (best_delta is not None and best_delta > _PTS_TOL_NS):
        return None
    return best


def set_obj_enc_context(ctx):
    """
    保留介面相容（舊版在此設定 GPU Object Encoder context）。
    現行截圖走 CPU（系統記憶體 RGBA + cv2），不需要 GPU 編碼器，故僅印出說明。
    """
    print("[INFO] [probes] 截圖機制已就緒（streammux 前 SysMem RGBA 緩存 + 依 PTS 時間戳對齊）")


def _crop_and_encode(n_frame, box):
    """
    從 RGBA 影像裁切指定方框並編碼為 JPEG bytes。

    參數：
        n_frame (ndarray): 來源 RGBA 影像（H, W, 4）
        box (tuple):       (left, top, width, height)，座標為畫面像素

    返回：
        bytes | None：成功回傳 JPEG bytes，框無效或編碼失敗回 None。
    """
    if n_frame is None: return None
    h, w = n_frame.shape[:2]
    left, top, bw, bh = box
    # 邊界裁切（clamp）到畫面範圍內，避免越界
    x1 = max(0, int(round(left)))
    y1 = max(0, int(round(top)))
    x2 = min(w, int(round(left + bw)))
    y2 = min(h, int(round(top + bh)))
    if x2 <= x1 or y2 <= y1: return None
    crop = n_frame[y1:y2, x1:x2]
    try:
        # appsink 來源是 RGBA，OpenCV 存檔需 BGR
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
        ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if not ok: return None
        return buf.tobytes()
    except Exception:
        return None


def _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg, n_frame):
    """
    處理「單一畫面內所有車輛物件」：累積 ROI 命中/車種票數、判斷進出方向、
    更新 OSD 標籤，並做車種截圖的 fallback（取面積最大的那次）。

    參數：
        gst_buffer:             目前的 GstBuffer
        frame_meta:             該畫面的 NvDsFrameMeta
        current_frame_objects:  本批次「仍存活」的物件 key 集合（供 housekeeping 判斷消失）
        pad_index (int):        哪一路 cam
        cfg (dict):             該路 cam 的設定
        n_frame (ndarray|None): 依 PTS 對到的該格 RGBA 畫面（截圖用，可能為 None）
    """
    cv_regions = cfg.get("cv_regions", {})
    movement_threshold = cfg.get("track_logic", {}).get("movement_threshold", 30)
    axis = cfg.get("track_logic", {}).get("axis", "y")   # "y"（上下）或 "x"（左右）
    up_left_is_out = cfg.get("track_logic", {}).get("up_left_is_out", True)  # 預設：往上/往左=OUT
    keep = cfg.get("keep_classes")                       # frozenset 或 None（None = 全收）
    save_ss = cfg.get("save_screenshot", False)

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try: obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration: break
        # 只處理車輛（PGIE）物件
        if obj_meta.unique_component_id != _UID_VEHICLE:
            l_obj = l_obj.next
            continue
        # 類別過濾：不在白名單內的 class_id 直接略過（不畫框 / 不計數 / 不寫 DB）
        if keep is not None and obj_meta.class_id not in keep:
            l_obj = l_obj.next
            continue
        obj_id = obj_meta.object_id
        if obj_id == -1:   # 未被追蹤器指派 ID，略過
            l_obj = l_obj.next
            continue
        unique_key = (pad_index, obj_id)
        current_frame_objects.add(unique_key)
        local_id = get_local_id(pad_index, obj_id)
        # 以「車框底邊中點」當代表點（cx, cy）：判斷 ROI 命中與移動方向
        cx = int(obj_meta.rect_params.left + (obj_meta.rect_params.width / 2))
        cy = int(obj_meta.rect_params.top + obj_meta.rect_params.height)

        # 第一次見到這台車：建立追蹤狀態
        if unique_key not in track_history:
            track_history[unique_key] = {
                "start_x": cx, "start_y": cy, "missing_frames": 0, "direction": "NA",
                "class_votes": Counter(), "plate_votes": Counter(),
                "last_frame_num": frame_meta.frame_num, "roi_hits": {},
                "best_class_jpg": None, "best_plate_jpg": None, "best_plate_area": 0,
                "fallback_class_jpg": None, "fallback_class_area": 0,
            }

        state = track_history[unique_key]
        state["missing_frames"] = 0                          # 這格有出現 → 消失計數歸零
        state["last_frame_num"] = frame_meta.frame_num
        r = obj_meta.rect_params

        # 代表點落在哪些 ROI 多邊形內 → 累積命中次數與車種票數
        for roi_name, polygon in cv_regions.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                state["roi_hits"][roi_name] = state["roi_hits"].get(roi_name, 0) + 1
                state["class_votes"][obj_meta.class_id] += 1

        # ⭐ 車種截圖 Fallback：沒有車牌時的備援，取「車框面積最大」的那次整車畫面
        if save_ss and n_frame is not None:
            veh_area = float(r.width) * float(r.height)
            if veh_area > state["fallback_class_area"]:
                jpg = _crop_and_encode(n_frame, (r.left, r.top, r.width, r.height))
                if jpg:
                    state["fallback_class_jpg"] = jpg
                    state["fallback_class_area"] = veh_area

        # 方向判斷：以首次出現位置為基準，所選軸位移超過門檻即定向
        #   axis="y"：delta>0 往下、delta<0 往上；axis="x"：delta>0 往右、delta<0 往左
        #   up_left_is_out=True（預設）：往上/往左=OUT、往下/往右=IN（與原本行為一致）
        #   up_left_is_out=False：整個對調（鏡頭反向時用）
        #   （DB 寫入的字串維持 "IN"/"OUT" 不變）
        if state["direction"] == "NA":
            delta = (cx - state["start_x"]) if axis == "x" else (cy - state["start_y"])
            if delta > movement_threshold:
                state["direction"] = "IN" if up_left_is_out else "OUT"
            elif delta < -movement_threshold:
                state["direction"] = "OUT" if up_left_is_out else "IN"

        # OSD：畫車框 + 標籤（ID + 車種）
        cls_id = obj_meta.class_id
        cls_name = CLASS_MAP.get(cls_id, f"Class_{cls_id}")
        color = get_class_color(cls_id)
        r.border_width = 4
        r.border_color.set(*color)
        r.has_bg_color = 0
        txt = obj_meta.text_params
        txt.display_text = f"ID:{local_id} {cls_name}"
        txt.font_params.font_name = "Serif Bold"
        txt.font_params.font_size = 14
        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(*color)
        text_h = int(14 * 1.4)
        txt.x_offset = max(0, int(r.left) + 0)
        txt.y_offset = max(0, int(r.top + r.height) - text_h - 10)
        l_obj = l_obj.next


def _post_frame_housekeeping(current_frame_objects):
    """
    每批次處理完後的雜務：
      1. 對「本批次沒再出現」的車輛累加消失計數；超過 cleanup_frames 即結算並清除。
      2. 每 30 秒印一次各路 FPS 報告。
      3. 依各路設定的間隔，定期把暫存事件 flush 進 SQLite。
    """
    global g_last_fps_print_time
    # 1. 消失偵測與結算
    missing_keys = set(track_history.keys()) - current_frame_objects
    for m_key in missing_keys:
        pad_index, obj_id = m_key
        cfg = SOURCE_CONFIGS.get(pad_index, {})
        track_history[m_key]["missing_frames"] += 1
        cleanup_frames = cfg.get("session", {}).get("cleanup_frames", 30)
        if track_history[m_key]["missing_frames"] >= cleanup_frames:
            _finalize_one(m_key, track_history[m_key], force=False)
            del track_history[m_key]
            if obj_id in local_id_maps[pad_index]: del local_id_maps[pad_index][obj_id]

    current_time = time.time()
    # 2. FPS 報告（每 30 秒）
    if current_time - g_last_fps_print_time >= 30:
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for sid, stats in sorted(fps_streams.items()):
            c_name = SOURCE_CONFIGS[sid].get("source_id", f"cam_{sid}")
            print(f" • {c_name.ljust(10)}: {stats['current_fps']:.2f} FPS")
        print("=" * 35 + "\n")
        g_last_fps_print_time = current_time

    # 3. 定期 flush 到 DB
    for pad_index, cfg in SOURCE_CONFIGS.items():
        flush_interval = cfg.get("session", {}).get("flush_interval_seconds", 30)
        if current_time - last_flush_times[pad_index] >= flush_interval:
            flush_pending_to_db(pad_index)
            last_flush_times[pad_index] = current_time


def _update_fps(pad_index):
    """以最近 30 個時間戳的滑動視窗估算該路的即時 FPS。"""
    if "timestamps" not in fps_streams[pad_index]: fps_streams[pad_index]["timestamps"] = deque(maxlen=30)
    now = time.time()
    q = fps_streams[pad_index]["timestamps"]
    q.append(now)
    if len(q) > 1: fps_streams[pad_index]["current_fps"] = (len(q) - 1) / (q[-1] - q[0])


def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    nvtracker.src 探針（TRACKER_MODE == "nvdcf" 時使用）。

    nvtracker 已完成追蹤並指派 object_id，這裡只需逐畫面、逐物件做後處理：
    依 PTS 取對應畫面 → _process_tracked_frame → housekeeping。
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try: frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration: break
        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        # ⭐ 依本格 PTS 取對應畫面（取代脆弱的計數器）
        n_frame = _lookup_frame_by_pts(pad_index, frame_meta.buf_pts)

        _update_fps(pad_index)
        _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg, n_frame)
        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


def boxmot_pgie_src_probe(pad, info, u_data):
    """
    pgie.src 探針（TRACKER_MODE != "nvdcf"，使用 BoxMOT 追蹤時）。

    流程：
      1. 蒐集 PGIE 在本格產生的所有偵測框，組成 dets，並把原始 obj_meta 從畫面移除
         （之後改用追蹤器輸出的框重新加回，確保畫面上的 ID 來自 BoxMOT）。
      2. 呼叫 BoxMOT 追蹤，得到帶 track_id 的框。
      3. 把追蹤結果以新的 obj_meta 加回畫面（unique_component_id = 車輛）。
      4. 依 PTS 取對應畫面 → _process_tracked_frame → housekeeping。
    """
    from logic.boxmot_adapter import track as boxmot_track
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try: frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration: break
        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        # ⭐ 依本格 PTS 取對應畫面（取代脆弱的計數器）
        n_frame = _lookup_frame_by_pts(pad_index, frame_meta.buf_pts)

        _update_fps(pad_index)

        # --- 1. 蒐集 PGIE 偵測框，並把原始 obj_meta 標記移除 ---
        keep = cfg.get("keep_classes")   # frozenset 或 None（None = 全收）
        dets_list = []
        obj_metas_to_remove = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try: obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration: break
            cls = int(obj_meta.class_id)
            # 類別過濾：不在白名單內的偵測框不餵給 BoxMOT（但仍要從畫面移除，避免殘留）
            if keep is not None and cls not in keep:
                obj_metas_to_remove.append(obj_meta)
                l_obj = l_obj.next
                continue
            try:
                # 優先用偵測器原始框（未經 tracker 平滑）
                det_box = obj_meta.detector_bbox_info.orgbbox_coords
                x1, y1 = float(det_box.left), float(det_box.top)
                x2, y2 = float(det_box.left + det_box.width), float(det_box.top + det_box.height)
            except Exception:
                r = obj_meta.rect_params
                x1, y1 = float(r.left), float(r.top)
                x2, y2 = float(r.left + r.width), float(r.top + r.height)
            conf = float(obj_meta.confidence) if obj_meta.confidence > 0 else 0.5
            dets_list.append([x1, y1, x2, y2, conf, cls])
            obj_metas_to_remove.append(obj_meta)
            l_obj = l_obj.next
        for om in obj_metas_to_remove: pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)

        # --- 2. BoxMOT 追蹤 ---
        dets = np.asarray(dets_list, dtype=np.float32) if dets_list else np.empty((0, 6), dtype=np.float32)
        tracks = boxmot_track(pad_index, dets, frame=None)

        # --- 3. 把追蹤結果加回畫面（供後續 SGIE / OSD / 截圖使用）---
        for tr in tracks:
            x1, y1, x2, y2 = float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3])
            tid, conf, cls = int(tr[4]), float(tr[5]), int(tr[6])
            new_obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            if new_obj is None: continue
            new_obj.unique_component_id = _UID_VEHICLE
            new_obj.class_id = cls
            new_obj.object_id = tid
            new_obj.confidence = conf
            new_obj.obj_label = CLASS_MAP.get(cls, f"Class_{cls}")
            r = new_obj.rect_params
            r.left, r.top = x1, y1
            r.width, r.height = max(1.0, x2 - x1), max(1.0, y2 - y1)
            r.border_width = 4
            r.has_bg_color = 0
            r.border_color.set(*get_class_color(cls))
            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj, None)

        # --- 4. 後處理（ROI/方向/車種票數/車種 fallback 截圖）---
        _process_tracked_frame(gst_buffer, frame_meta, current_frame_objects, pad_index, cfg, n_frame)
        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


def expand_plate_probe(pad, info, u_data):
    """
    sgie_plate.src 探針：把車牌框（PLATE）四邊各往外擴 10%，
    避免後續字元偵測/裁切時車牌邊緣被切掉；擴框後夾回畫面範圍內。
    同時關閉車牌框上的文字標籤（之後在 assemble 階段才填車牌字串）。
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try: frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration: break
        frame_w, frame_h = frame_meta.source_frame_width, frame_meta.source_frame_height
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try: obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration: break
            if obj_meta.unique_component_id == _UID_PLATE:
                rect = obj_meta.rect_params
                w, h = rect.width, rect.height
                dw, dh = w * 0.10, h * 0.10
                rect.left -= dw; rect.top -= dh
                rect.width += 2.0 * dw; rect.height += 2.0 * dh
                # 夾回畫面範圍
                rect.left = max(0.0, rect.left); rect.top = max(0.0, rect.top)
                if rect.left + rect.width > frame_w: rect.width = frame_w - rect.left
                if rect.top + rect.height > frame_h: rect.height = frame_h - rect.top
                rect.border_width = 3
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                obj_meta.text_params.set_bg_clr = 0
                obj_meta.text_params.font_params.font_size = 0
            l_obj = l_obj.next
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def _frame_chars_to_string(chars, char_nms_iou=0.5):
    """
    把單台車的字元偵測結果組成車牌字串。

    步驟：先用 NMS 去除重疊字元框，再依框中心 x 由左到右排序，最後查表轉成字元。

    參數：
        chars (list[dict]): 每個字元含 x1,y1,x2,y2,score,cls_id
        char_nms_iou (float): NMS 的 IoU 門檻

    返回：
        str：組好的車牌字串（可能為空字串）
    """
    if not chars: return ""
    x1 = np.array([c["x1"] for c in chars])
    y1 = np.array([c["y1"] for c in chars])
    x2 = np.array([c["x2"] for c in chars])
    y2 = np.array([c["y2"] for c in chars])
    scores = np.array([c["score"] for c in chars])
    cls_ids = np.array([c["cls_id"] for c in chars])
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), 0.0, char_nms_iou)
    if idxs is None or len(idxs) == 0: return ""
    idxs = np.array(idxs).flatten()
    fx1, fx2, fcls = x1[idxs], x2[idxs], cls_ids[idxs]
    order = np.argsort((fx1 + fx2) / 2.0)   # 依水平位置由左到右
    return "".join(NUM_MAP.get(int(fcls[i]), "") for i in order)


def assemble_plate_probe(pad, info, u_data):
    """
    sgie_num.src 探針：本格已含「車輛框 / 車牌框 / 字元框」三種 meta。

    主要工作：
      1. 蒐集本格的 vehicles / plates / chars 三類物件。
      2. 用「中心點落在車框內 + 重疊面積最大」把字元與車牌各自配對到車輛。
      3. 把字元組成車牌字串，累積到該車的 plate_votes（多格投票，最後取最高票）。
      4. ⭐ 截圖：以本格 PTS 對到的同一張畫面，裁切
           - 車牌框 → best_plate_jpg
           - 車輛框∪車牌框（再往下補一點）→ best_class_jpg
         僅在「車牌面積比之前更大」時更新，確保留下最清晰的一張。
         以 PTS 對齊同一格，車牌小框才不會因時間差而裁到背景。
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try: frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration: break
        pad_idx = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_idx, {})
        save_ss = cfg.get("save_screenshot", False)
        frame_w, frame_h = frame_meta.source_frame_width, frame_meta.source_frame_height

        # ⭐ 依本格 PTS 取對應畫面（與偵測框同一格，車牌小框才不會錯位）
        n_frame = _lookup_frame_by_pts(pad_idx, frame_meta.buf_pts) if save_ss else None

        # --- 1. 蒐集本格三類物件 ---
        vehicles = {}
        plates = []
        chars = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try: obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration: break
            uid = obj.unique_component_id
            r = obj.rect_params
            x1, y1 = float(r.left), float(r.top)
            x2, y2 = x1 + float(r.width), y1 + float(r.height)
            if uid == _UID_VEHICLE:
                v_id = obj.object_id
                if v_id != -1: vehicles[v_id] = {"obj": obj, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            elif uid == _UID_PLATE:
                plates.append({"obj": obj, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": float(r.width) * float(r.height)})
            elif uid == _UID_CHAR:
                # 字元框本身不畫出來（避免畫面雜亂），只取座標供組字
                r.border_width = 0; r.has_bg_color = 0
                txt = obj.text_params; txt.set_bg_clr = 0; txt.font_params.font_size = 0; txt.display_text = ""
                chars.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(obj.confidence), "cls_id": int(obj.class_id)})
            l_obj = l_obj.next

        active_vehicles = [{"v_id": v_id, "x1": v["x1"], "y1": v["y1"], "x2": v["x2"], "y2": v["y2"]} for v_id, v in vehicles.items()]

        # --- 2a. 字元 → 車輛 配對（中心點落在車框內、取重疊面積最大者）---
        vehicle_chars = {}
        for c in chars:
            ccx, ccy = (c["x1"] + c["x2"]) / 2.0, (c["y1"] + c["y2"]) / 2.0
            best_vid, best_overlap = None, 0.0
            for v in active_vehicles:
                if not (v["x1"] <= ccx <= v["x2"] and v["y1"] <= ccy <= v["y2"]): continue
                ix1, iy1 = max(c["x1"], v["x1"]), max(c["y1"], v["y1"])
                ix2, iy2 = min(c["x2"], v["x2"]), min(c["y2"], v["y2"])
                ov = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if ov > best_overlap: best_overlap, best_vid = ov, v["v_id"]
            if best_vid is not None: vehicle_chars.setdefault(best_vid, []).append(c)

        # --- 2b. 車牌 → 車輛 配對（同樣規則）---
        plate_to_vehicle = {}
        for pi, p in enumerate(plates):
            pcx, pcy = (p["x1"] + p["x2"]) / 2.0, (p["y1"] + p["y2"]) / 2.0
            best_vid, best_overlap = None, 0.0
            for v in active_vehicles:
                if not (v["x1"] <= pcx <= v["x2"] and v["y1"] <= pcy <= v["y2"]): continue
                ix1, iy1 = max(p["x1"], v["x1"]), max(p["y1"], v["y1"])
                ix2, iy2 = min(p["x2"], v["x2"]), min(p["y2"], v["y2"])
                ov = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if ov > best_overlap: best_overlap, best_vid = ov, v["v_id"]
            plate_to_vehicle[pi] = best_vid

        # --- 3. 車牌字串投票（每格一票，累積到該車）---
        for v_id, chars_for_v in vehicle_chars.items():
            v_key = (pad_idx, v_id)
            if v_key not in track_history: continue
            plate_str = _frame_chars_to_string(chars_for_v, char_nms_iou=0.5)
            if plate_str:
                state = track_history[v_key]
                if "plate_votes" not in state: state["plate_votes"] = Counter()
                state["plate_votes"][plate_str] += 1

        # --- 4. 車牌 OSD 標籤 + ⭐ 截圖（車牌框 / 車種聯集框）---
        for pi, p in enumerate(plates):
            plate_obj = p["obj"]
            v_id = plate_to_vehicle.get(pi)
            plate_str = ""
            if v_id is not None and v_id in vehicle_chars: plate_str = _frame_chars_to_string(vehicle_chars[v_id], char_nms_iou=0.5)
            txt = plate_obj.text_params; r = plate_obj.rect_params
            if plate_str:
                # 在車牌框下方標出車牌字串
                txt.display_text = plate_str; txt.font_params.font_name = "Serif Bold"; txt.font_params.font_size = 13
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0); txt.set_bg_clr = 1; txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                txt.x_offset = int(r.left); txt.y_offset = max(0, int(r.top + r.height))
            else:
                txt.set_bg_clr = 0; txt.font_params.font_size = 0

            # 只有「成功配對車輛 + 有組出車牌字串 + 有對到畫面」才截圖
            if save_ss and v_id is not None and plate_str and n_frame is not None:
                v_key = (pad_idx, v_id)
                if v_key in track_history:
                    state = track_history[v_key]
                    plate_area = p["area"]
                    # 只在「車牌面積比之前更大」時更新 → 留下最清晰的一張
                    if plate_area > state.get("best_plate_area", 0):
                        # (a) 車牌截圖
                        pl, pt = float(p["x1"]), float(p["y1"])
                        pw, ph = max(1.0, float(p["x2"] - p["x1"])), max(1.0, float(p["y2"] - p["y1"]))
                        plate_jpg = _crop_and_encode(n_frame, (pl, pt, pw, ph))
                        if plate_jpg: state["best_plate_jpg"] = plate_jpg

                        # (b) 車種截圖：車輛框 ∪ 車牌框，再往下補一點車尾/車頭
                        class_box = None
                        veh_entry = vehicles.get(v_id)
                        if veh_entry is not None:
                            o_x1, o_y1 = veh_entry["x1"], veh_entry["y1"]
                            o_x2, o_y2 = veh_entry["x2"], veh_entry["y2"]
                            plate_h = p["y2"] - p["y1"]
                            bottom_pad = plate_h * _PLATE_BOTTOM_PAD_RATIO
                            u_x1, u_y1 = min(float(o_x1), p["x1"]), min(float(o_y1), p["y1"])
                            u_x2, u_y2 = max(float(o_x2), p["x2"]), max(float(o_y2), p["y2"]) + bottom_pad
                            u_x1, u_y1 = max(0.0, u_x1), max(0.0, u_y1)
                            u_x2, u_y2 = min(float(frame_w), u_x2), min(float(frame_h), u_y2)
                            class_box = (u_x1, u_y1, max(1.0, u_x2 - u_x1), max(1.0, u_y2 - u_y1))

                        if class_box is not None:
                            class_jpg = _crop_and_encode(n_frame, class_box)
                            if class_jpg: state["best_class_jpg"] = class_jpg

                        state["best_plate_area"] = plate_area
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def per_cam_osd_probe(pad, info, pad_index):
    """
    每路 nvosd.sink 探針（demux 之後、每路各一個）：在畫面左上角疊加即時 FPS 文字。

    參數：
        pad, info:        GStreamer probe 標準參數
        pad_index (int):  哪一路 cam（由 setup_cam_branch 以 lambda 帶入）
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    cfg = SOURCE_CONFIGS.get(pad_index)
    if not cfg: return Gst.PadProbeReturn.OK
    show_fps = cfg.get("display", {}).get("show_fps_overlay", True)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try: frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration: break
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 0; display_meta.num_lines = 0; display_meta.num_rects = 0; display_meta.num_circles = 0
        if show_fps and pad_index in fps_streams:
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = f"FPS: {fps_streams[pad_index]['current_fps']:.1f}"
            txt_params.x_offset = 5; txt_params.y_offset = 5
            txt_params.font_params.font_name = "Serif Bold"; txt_params.font_params.font_size = 25
            txt_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0); txt_params.set_bg_clr = 1; txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK
