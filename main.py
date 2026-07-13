#!/usr/bin/env python3
"""
================================================================================
 main.py — DeepStream 7.1 多路車牌辨識 (LPR) 主程式
================================================================================
整體 pipeline：

  [每路 source] nvurisrcbin ─ tee ┬─► (主推論) nvstreammux
                                   └─► (截圖)   queue→nvvideoconvert→appsink

  nvstreammux → q1 → nvdspreprocess → q2 → pgie(車輛) → q3
              → q_sgie_plate → sgie_plate(車牌) → q_sgie_num → sgie_num(字元)
              → q_analytics → nvdsanalytics → q4 → nvstreamdemux
              → (每路) queue → nvvideoconvert → nvdsosd → 顯示/存檔

RTSP 斷線重連（只作用於即時串流路，檔案來源不受影響）：
  第一層：nvurisrcbin 內建 rtsp-reconnect（5 秒間隔、無限重試），處理「有斷線錯誤」情況。
  第二層：看門狗每 10 秒檢查各 RTSP 路最後吐幀時間，卡死 60 秒即單路重啟（NULL→PLAYING），
          其他路與截圖分支完全不受影響；重啟後 cb_newpad 把新 pad 接回既有 tee。
  兩層都持續到 EOS 為止：按 'q' / SIGINT / SIGTERM 觸發 EOS 後看門狗立即停止，
  無頭（headless / systemd）模式用訊號即可優雅收尾。

截圖設計重點：
  截圖影像在「streammux 之前」就用 tee 分流到 appsink 取得乾淨 RGBA（系統記憶體），
  並以該幀 PTS 為鍵存入 probes.g_frame_buffer。各 probe 再用 frame_meta.buf_pts
  找回「同一格」畫面來裁切。如此可避開 demux 後 per-source 取像的限制，
  並以 PTS 對齊，避免計數器在丟幀時錯位（車牌小框尤其敏感）。

操作：執行後在終端機按 'q' 可優雅送出 EOS、等待影片封裝完成後結算退出。
================================================================================
"""
import sys
import time
import termios
import tty
import signal
import numpy as np
# 啟用新版 nvstreammux：多路檔案來源時某一路先 EOS 不拖慢其餘來源。
# 必須在 import gi 之前，GStreamer 載入 nvstreammux 外掛時才讀得到。
import os
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "no")
# GLib/GIO 建立網路（RTSP）連線時會呼叫系統 libproxy 偵測 proxy。在 conda 環境下，
# conda 的 libstdc++ 與系統 libunwind ABI 不相容，libproxy 拋例外時無法正常 unwind
# 而導致行程 abort。改用 GIO 內建 dummy proxy resolver 完全繞過 libproxy。
# 須在匯入 gi 之前設定；系統 Python 不受影響，setdefault 也不覆蓋外部既有設定。
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds

from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
    INFER_SEC_PLATE_CONFIG, INFER_SEC_NUM_CONFIG,
    TRACKER_MODE,
)
from logic.state_db import initialize_state_managers, force_finalize_all, fps_streams
from logic.pipeline import (
    cb_decodebin_child_added, make_elm, resolve_tracker_lib, _safe_set,
    _build_display_sink, setup_cam_branch,
)
from logic.probes import (
    tracker_src_pad_buffer_probe,
    boxmot_pgie_src_probe,
    expand_plate_probe,
    assemble_plate_probe,
    per_cam_osd_probe,
    set_obj_enc_context,
    store_frame_for_pad,    # ⭐ appsink 用：把幀連同 PTS 存入緩存
)

# ---- 全域物件（供 keyboard / bus / 退出流程共用）----
g_loop          = None
g_pipeline      = None
g_eos_triggered = False
g_obj_enc_ctx   = None

# ---- RTSP 斷線重連 / 看門狗（只針對「即時串流」來源，檔案來源不適用）----
# 第一層：nvurisrcbin 內建 rtsp-reconnect（無限重試），涵蓋「有斷線錯誤」的情況。
# 第二層：看門狗定期檢查各路最後吐幀時間，涵蓋「無錯誤但卡死不吐幀」的情況（單路重啟）。
# 兩層都只到 EOS 為止：按 Q / SIGINT / SIGTERM 觸發 EOS 後，看門狗立即停止，不干擾收尾。
g_sources       = {}   # pad_index -> {"src","cam","streammux","cfg","rebuilds"}（只收錄 RTSP 路）
g_stall_notify  = {}   # pad_index -> 具名卡幀通知累計次數（恢復吐幀即歸零）
g_restart_counts = {}  # pad_index -> 看門狗連續重啟/重建次數（恢復吐幀即歸零）
g_last_restart = {}    # pad_index -> 上次重啟時間戳（防連環重啟）
WATCHDOG_STALL_SEC = 60    # 連續幾秒沒吐幀 → 判定卡死
WATCHDOG_GRACE_SEC = 60    # 重啟後寬限幾秒（期間不再判定）
WATCHDOG_CHECK_SEC = 10    # 每幾秒檢查一次
WATCHDOG_NOTIFY_SEC = 15   # 卡幀超過此秒數即開始「具名」回報（早於重啟門檻，方便定位是哪一路）
WATCHDOG_MIN_FPS = 1.0     # 滴幀偵測門檻：觀察窗內平均 FPS 低於此值視同卡死。堵住「內建重連
                           # 半成功、每次滴進幾幀把 idle 重置，導致看門狗永遠不觸發」的漏洞
WATCHDOG_REBUILD_AFTER = 3 # 同一路連續整顆重啟 N 次仍未恢復 → 升級為「移除元件、全新重建」
                           # （全新 RTSP session，等效手動關掉播放器重開，可跳出半成功迴圈）


def force_quit_loop():
    """送出 EOS 後的逾時保險：等待影片封裝逾時就強制結束 MainLoop。"""
    global g_loop
    print("\n[WARNING] 等待影片封裝逾時，強制退出管線！")
    if g_loop and g_loop.is_running(): g_loop.quit()
    return False


def keyboard_cb(fd, condition):
    """鍵盤監看：按 'q'/'Q' → 送出 EOS 並啟動 8 秒逾時保險。"""
    global g_eos_triggered, g_pipeline, g_loop
    ch = sys.stdin.read(1)
    if ch in ('q', 'Q') and not g_eos_triggered:
        g_eos_triggered = True
        print("\n[INFO] 收到 'Q' 鍵，正在安全發送 EOS 訊號 (等待影片寫入)...")
        if g_pipeline:
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        return False
    return True


def bus_call(bus, message, loop):
    """GStreamer Bus 訊息處理：EOS 正常退出；RTSP 類錯誤容忍不退，其餘嚴重錯誤退出。"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] 影像串流結束 (EOS 處理完畢)，準備安全退出...")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        err_msg = str(err).lower()
        # RTSP 串流抖動/中斷不致命：保持運行等待重連
        if ("rtsp" in err_msg or "timeout" in err_msg or "resource not found" in err_msg or "could not read" in err_msg):
            print(f"[WARNING] RTSP 串流不穩或中斷: {err}。系統保持運行，等待自動重連...")
        else:
            print(f"[ERROR] 嚴重管線錯誤: {err}: {debug}")
            loop.quit()
    return True


# ==========================================
# ⭐ 核心：appsink 回調，將 SysMem RGBA 連同「PTS」存入 g_frame_buffer
#    不再用計數器；用每格自帶的 buf.pts 當對齊鍵，丟幀也不會錯位。
# ==========================================
def cb_appsink_sample(appsink, pad_index):
    """
    appsink 每收到一格畫面就觸發：把 RGBA 影像複製成 numpy，並連同其 PTS 存入緩存。

    參數：
        appsink:          觸發此回調的 appsink 元件
        pad_index (int):  哪一路 cam（connect 時帶入）
    """
    sample = appsink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                caps = sample.get_caps()
                if caps:
                    struct = caps.get_structure(0)
                    ok_w, w = struct.get_int("width")
                    ok_h, h = struct.get_int("height")
                    if ok_w and ok_h:
                        # 直接從 mapinfo.data 建立 numpy array (RGBA) 並複製一份
                        # （複製是必要的：unmap 之後原記憶體即失效）
                        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 4)).copy()
                        # ⭐ 以該幀的 PTS 為鍵存入（環形緩存，內部自動限長）
                        store_frame_for_pad(pad_index, buf.pts, frame)
            finally:
                buf.unmap(mapinfo)
    return Gst.FlowReturn.OK


# ==========================================
# ⭐ 核心：cb_newpad，在 streammux 前分流出截圖分支
# ==========================================
def _drain_pad_to_fakesink(pipeline, src_pad):
    """把不需要的 pad（音訊 / 多餘視訊）接到 fakesink 消化，避免未連結 pad 反壓卡整路。"""
    fs = Gst.ElementFactory.make("fakesink", None)   # None = 自動命名，避免重啟後撞名
    if fs is None:
        return
    fs.set_property("sync", False)
    fs.set_property("async", False)
    pipeline.add(fs)
    fs.sync_state_with_parent()
    src_pad.link(fs.get_static_pad("sink"))


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    nvurisrcbin 解碼出視訊 pad 時觸發。用 tee 把解碼後畫面分成兩路：
      分支 A：接到 nvstreammux（主推論線）。
      分支 B：（若該路啟用截圖）queue→nvvideoconvert(RGBA,系統記憶體)→appsink。

    為何在 streammux 之前分流：這裡拿到的是「乾淨、未疊 OSD、且每路獨立」的畫面，
    且不受 demux 後 per-source 取像限制；appsink 走 leaky/drop，不會反壓主推論。

    重啟支援（看門狗單路重啟後會再次觸發本回呼）：
      tee 與其下游（streammux 連結、截圖分支）掛在 pipeline 上、不隨 source 重啟消失；
      source NULL→PLAYING 後只有「decoder → tee.sink」這條斷掉，
      因此偵測到既有 tee 時，把新的 decoder pad 直接接回 tee.sink 即完成復原。
    """
    caps = decoder_src_pad.get_current_caps()
    if not caps: caps = decoder_src_pad.query_caps()
    pad_index = data["pad_index"]
    streammux = data["streammux"]
    pipeline = data["pipeline"]
    cfg = data["cfg"]

    # 非影像流（音訊等）→ fakesink 消化，避免未連結 pad 造成反壓
    if caps.get_structure(0).get_name().find("video") == -1:
        _drain_pad_to_fakesink(pipeline, decoder_src_pad)
        return

    # ---- 重啟路徑：tee 已存在（單路重啟後重新吐 pad）→ 直接接回 ----
    tee = pipeline.get_by_name(f"tee-src-{pad_index}")
    if tee is not None:
        tee_sink = tee.get_static_pad("sink")
        if tee_sink.is_linked():
            # tee 已被接著（例如來源吐出第二條視訊流）→ 多餘的導 fakesink
            _drain_pad_to_fakesink(pipeline, decoder_src_pad)
            return
        decoder_src_pad.link(tee_sink)
        cam = cfg.get("source_id", f"cam{pad_index}")
        print(f"[WATCHDOG] {cam} 已重新接回主線（decoder → tee），恢復辨識")
        return

    # ---- 首次路徑：建立 tee 分流（原邏輯）----
    sinkpad = streammux.get_request_pad(f"sink_{pad_index}")
    if not sinkpad.is_linked():
        # 建立 tee 分流
        tee = make_elm("tee", f"tee-src-{pad_index}")
        pipeline.add(tee)
        tee.sync_state_with_parent()  # ⭐ 關鍵：同步狀態，避免 Pipeline 卡死

        decoder_src_pad.link(tee.get_static_pad("sink"))

        # 分支 A: 主推論 (連接到 streammux)
        tee_src_main = tee.get_request_pad("src_%u")
        tee_src_main.link(sinkpad)

        # 分支 B: 截圖專用 (如果啟用)
        if cfg.get("save_screenshot", False):
            q_ss = make_elm("queue", f"q-ss-src-{pad_index}")
            q_ss.set_property("max-size-buffers", 4)
            q_ss.set_property("leaky", 2)   # 丟舊幀，不反壓主推論

            conv_ss = make_elm("nvvideoconvert", f"conv-ss-src-{pad_index}")
            conv_ss.set_property("nvbuf-memory-type", 0)

            caps_ss = make_elm("capsfilter", f"caps-ss-src-{pad_index}")
            # 系統記憶體 RGBA（無 memory:NVMM）→ CPU 可直接 map 出 numpy
            caps_ss.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))

            sink_ss = make_elm("appsink", f"appsink-ss-{pad_index}")
            sink_ss.set_property("sync", False)
            sink_ss.set_property("async", False)
            sink_ss.set_property("emit-signals", True)
            sink_ss.set_property("max-buffers", 10)
            sink_ss.set_property("drop", True)
            sink_ss.connect("new-sample", cb_appsink_sample, pad_index)

            for elm in [q_ss, conv_ss, caps_ss, sink_ss]:
                pipeline.add(elm)
                elm.sync_state_with_parent()  # ⭐ 關鍵：同步狀態

            tee_src_ss = tee.get_request_pad("src_%u")
            tee_src_ss.link(q_ss.get_static_pad("sink"))
            q_ss.link(conv_ss)
            conv_ss.link(caps_ss)
            caps_ss.link(sink_ss)
            print(f"[INFO] cam={pad_index} 已建立 streammux 前截圖分支 (SysMem RGBA)")


def _enlarge_queue(q, max_buffers=400):
    """放大 queue 容量（buffers 上限放大，bytes/time 不限），緩衝推論前後的速差。"""
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


def _create_source_element(pad_index, cfg, streammux, name):
    """
    建立並設定一顆 nvurisrcbin（URI、RTSP 重連屬性、pad-added / child-added 訊號）。
    不加入 pipeline、不設狀態——由呼叫端處理。初次建立與看門狗「重建」共用，設定保證一致。
    """
    source = make_elm("nvurisrcbin", name)
    source.set_property("uri", cfg["source"])
    if not cfg.get("is_file_source", False):
        # source-id：讓 nvurisrcbin 內建訊息（如 Resetting source N）能辨識是哪一路，不設會顯示 -1
        _safe_set(source, "source-id", pad_index)
        _safe_set(source, "rtsp-reconnect-interval", 5)    # 連續 5 秒收不到資料就重連
        _safe_set(source, "rtsp-reconnect-attempts", -1)   # -1 = 無限重試，永不放棄
        _safe_set(source, "select-rtp-protocol", 4)        # 4 = 強制 TCP
        _safe_set(source, "latency", 200)                  # 抖動緩衝 200ms
        _safe_set(source, "udp-buffer-size", 2000000)
    # ⭐ 傳入更多參數給 cb_newpad（streammux/pad_index/pipeline/cfg；重啟後接回既有 tee 需要）
    source.connect("pad-added", cb_newpad, {
        "streammux": streammux,
        "pad_index": pad_index,
        "pipeline": g_pipeline,
        "cfg": cfg
    })
    # 內部 rtspsrc 的 TCP / 逾時調校（nvurisrcbin 用 child-added 遞迴抓，取代 source-setup）
    source.connect("child-added", cb_decodebin_child_added, None)
    return source


def _restart_one_source(pad_index):
    """
    單獨重啟某一路 nvurisrcbin（看門狗判定卡死時由 GLib.idle_add 在主線程呼叫）。

    LPR 的 tee 與其下游（streammux 連結、截圖分支）掛在 pipeline 上，不隨 source 消失，
    因此只需：source 設 NULL → 等狀態確實切換 → 重設 PLAYING。
    重啟後 nvurisrcbin 重連來源、重新吐 pad → cb_newpad 偵測到既有 tee 直接接回。
    不動 streammux 的 sink pad、不動截圖分支，其他路完全不受影響。
    """
    info = g_sources.get(pad_index)
    if not info:
        return False
    if g_eos_triggered:      # 已在收尾流程 → 不再重啟
        return False
    src, cam = info["src"], info["cam"]
    g_restart_counts[pad_index] = g_restart_counts.get(pad_index, 0) + 1
    n = g_restart_counts[pad_index]
    rebuild = n > WATCHDOG_REBUILD_AFTER   # 前 N 次：整顆重啟；之後每次：移除元件全新重建
    print(f"[WATCHDOG] {'重建' if rebuild else '重啟'} {cam}（pad={pad_index}）— 連續第 {n} 次"
          + (f"（前 {WATCHDOG_REBUILD_AFTER} 次重啟未恢復 → 升級為移除元件、建立全新 RTSP session）" if rebuild and n == WATCHDOG_REBUILD_AFTER + 1 else ""))
    try:
        # 該路 source 設 NULL，等狀態確實切換（tee 與下游掛在 pipeline 上，不受影響）
        src.set_state(Gst.State.NULL)
        src.get_state(Gst.CLOCK_TIME_NONE)

        if rebuild:
            # 升級路徑：把舊元件整顆移出 pipeline，建立「全新」nvurisrcbin。
            # 全新元件 = 全新 rtspsrc / 全新 TCP 連線 / 內部狀態歸零，
            # 等效手動關掉播放器重開，可跳出「半成功重連迴圈」等內部卡死狀態。
            # 新元件吐 pad → cb_newpad 偵測到既有 tee，直接接回 decoder → tee.sink。
            g_pipeline.remove(src)
            info["rebuilds"] = info.get("rebuilds", 0) + 1
            new_name = f"uri-decode-bin-{pad_index}-r{info['rebuilds']}"
            new_src = _create_source_element(pad_index, info["cfg"], info["streammux"], new_name)
            g_pipeline.add(new_src)
            new_src.sync_state_with_parent()
            info["src"] = new_src
            g_last_restart[pad_index] = time.time()
            print(f"[WATCHDOG] {cam} 已重建為全新元件（{new_name}），等待重新連線...")
        else:
            # 一般路徑：重新 PLAYING → nvurisrcbin 重連、重新吐 pad → cb_newpad 接回既有 tee
            src.set_state(Gst.State.PLAYING)
            g_last_restart[pad_index] = time.time()
            print(f"[WATCHDOG] {cam} 已送出重啟，等待重新連線...")
    except Exception as e:
        print(f"[WATCHDOG] {'重建' if rebuild else '重啟'} {cam} 發生例外: {e}")
    return False   # 給 idle_add 用，只跑一次


def _watchdog_check():
    """
    每 WATCHDOG_CHECK_SEC 秒檢查各「RTSP 路」最後吐幀時間，卡死超過門檻就單路重啟。

    只監控 g_sources 內的路（建立來源時只收錄 RTSP，檔案來源播完不吐幀是正常現象，
    絕不能重啟——否則影片會從頭重播、DB 重複計數）。
    EOS 觸發（按 Q / SIGINT / SIGTERM）後回傳 False 停止本 timer，不干擾收尾封裝。
    """
    if g_eos_triggered:
        print("[WATCHDOG] 偵測到 EOS 收尾流程，看門狗停止")
        return False
    now = time.time()
    for pad_index, info in list(g_sources.items()):
        # 重啟寬限期內不判定
        if now - g_last_restart.get(pad_index, 0) < WATCHDOG_GRACE_SEC:
            continue
        stats = fps_streams.get(pad_index, {})
        ts = stats.get("timestamps")
        if not ts:
            continue   # 還沒收過任何幀（剛啟動/首連中），交給 nvurisrcbin 內建重連
        idle = now - ts[-1]
        cam = info['cam']

        # --- 滴幀偵測：內建重連「半成功」時每次會滴進幾幀，idle 一直被重置，
        #     單看 idle 永遠不會達標。改看觀察窗平均 FPS：最近 len(ts) 幀共花 span 秒，
        #     健康串流 30 幀約 1-2 秒；若 span 已超過卡死門檻仍湊不滿，就是滴幀卡死。
        span = now - ts[0]
        rate = (len(ts) / span) if span > 0 else 999.0
        trickle = (span >= WATCHDOG_STALL_SEC and rate < WATCHDOG_MIN_FPS)
        stalled = (idle >= WATCHDOG_STALL_SEC) or trickle

        # --- 具名狀態回報：一眼看出是哪一路在卡、卡多久、通知第幾次 ---
        if idle >= WATCHDOG_NOTIFY_SEC or trickle:
            g_stall_notify[pad_index] = g_stall_notify.get(pad_index, 0) + 1
            desc = (f"已 {idle:.0f}s 無新幀" if idle >= WATCHDOG_NOTIFY_SEC
                    else f"滴幀中（近 {span:.0f}s 平均僅 {rate:.2f} FPS）")
            print(f"[{cam}] {desc}（nvurisrcbin 內建重連中）— 第 {g_stall_notify[pad_index]} 次通知")
        elif g_stall_notify.get(pad_index, 0) > 0:
            # 從卡幀狀態恢復 → 回報並歸零計數
            print(f"[{cam}] ✅ 已恢復吐幀（先前通知 {g_stall_notify[pad_index]} 次、"
                  f"看門狗重啟/重建 {g_restart_counts.get(pad_index, 0)} 次）")
            g_stall_notify[pad_index] = 0
            g_restart_counts[pad_index] = 0
            info["rebuilds"] = 0

        if stalled:
            reason = f"已 {idle:.0f} 秒無新幀" if idle >= WATCHDOG_STALL_SEC else f"滴幀（近 {span:.0f}s 平均 {rate:.2f} FPS）"
            print(f"[WATCHDOG] {cam}（pad={pad_index}）{reason}，判定卡死 → 單路處置")
            GLib.idle_add(_restart_one_source, pad_index)
    return True   # 回 True 讓 timer 持續


def _init_obj_encoder_if_needed():
    """
    截圖走 CPU（系統記憶體 RGBA + cv2）路徑，不需要 GPU Object Encoder context。
    此函式僅印出說明並把 obj_enc context 設為 None（保留舊介面）。
    """
    need_ss = any(cfg.get("save_screenshot", False) for cfg in SOURCE_CONFIGS.values())
    if need_ss: print("[INFO] 截圖走 CPU（系統記憶體 RGBA + cv2）路徑，不建立 GPU Object Encoder context")
    else: print("[INFO] 無 cam 啟用截圖")
    set_obj_enc_context(None)
    return None


def main():
    global g_loop, g_pipeline, g_eos_triggered, g_obj_enc_ctx

    # ---- 追蹤器初始化（BoxMOT 模式需先為各路建立 tracker）----
    if TRACKER_MODE == "nvdcf":
        print("[INFO] 初始化 DeepStream LPR 三層架構：PGIE → NvDCF → SGIE plate → SGIE num → Analytics")
    else:
        print(f"[INFO] 初始化 DeepStream LPR 三層架構：PGIE → {TRACKER_MODE} (BoxMOT) → SGIE plate → SGIE num → Analytics")
        from logic.boxmot_adapter import initialize_boxmot_trackers
        initialize_boxmot_trackers()

    Gst.init(None)
    g_obj_enc_ctx = _init_obj_encoder_if_needed()
    g_pipeline = Gst.Pipeline.new("lpr-pipeline")
    num_sources = len(SOURCE_CONFIGS)
    show_window = any(cfg.get("display", {}).get("show_window", True) for cfg in SOURCE_CONFIGS.values())

    # ---- nvstreammux：把多路畫面批成一個 batch 餵給推論 ----
    streammux = make_elm("nvstreammux", "Stream-muxer")
    streammux.set_property("batch-size", num_sources)  # 新舊版 mux 皆支援

    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改用 config_mux.txt
        _mux_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mux.txt")
        if os.path.exists(_mux_cfg):
            streammux.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，新版 mux 將用內建預設值")
    else:
        # 舊版 mux：維持原本設定
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batched-push-timeout", 10000)
        streammux.set_property("live-source", 1)
        streammux.set_property("nvbuf-memory-type", 0)
    g_pipeline.add(streammux)

    # ---- 各路 source：nvurisrcbin（uridecodebin 沒有重連能力），pad-added 時在 cb_newpad 內分流 ----
    for pad_index, cfg in SOURCE_CONFIGS.items():
        # 建立 + 屬性 + 訊號統一走 _create_source_element（初次與看門狗「重建」共用，設定保證一致）
        source = _create_source_element(pad_index, cfg, streammux, f"uri-decode-bin-{pad_index}")
        g_pipeline.add(source)

        is_live = not cfg.get("is_file_source", False)
        if is_live:
            print(f"[INFO] {cfg.get('source_id', pad_index)} 為即時串流：啟用自動重連（5s 間隔、無限重試）")
            # --- 第二層防護：看門狗只登記 RTSP 路（檔案播完不吐幀是正常現象，不能重啟）---
            # streammux / cfg 供「重建」時建立全新元件用
            g_sources[pad_index] = {
                "src": source, "cam": cfg.get("source_id", f"cam{pad_index}"),
                "streammux": streammux, "cfg": cfg, "rebuilds": 0,
            }

    # ---- 中段 queue ----
    q1, q2, q3 = make_elm("queue", "q1"), make_elm("queue", "q2"), make_elm("queue", "q3")
    q_sgie_plate = make_elm("queue", "q_sgie_plate")
    q_sgie_num = make_elm("queue", "q_sgie_num")
    q_analytics = make_elm("queue", "q_analytics")
    q4 = make_elm("queue", "q4")
    _enlarge_queue(q_sgie_plate, 400)
    _enlarge_queue(q_sgie_num, 400)
    _enlarge_queue(q_analytics, 200)

    # ---- 推論/分析元件 ----
    preprocess = make_elm("nvdspreprocess", "preprocess")
    preprocess.set_property("config-file", PREPROCESS_CONFIG)
    pgie = make_elm("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)
    pgie.set_property("input-tensor-meta", True)
    sgie_plate = make_elm("nvinfer", "secondary-plate")
    sgie_plate.set_property("config-file-path", INFER_SEC_PLATE_CONFIG)
    sgie_num = make_elm("nvinfer", "secondary-num")
    sgie_num.set_property("config-file-path", INFER_SEC_NUM_CONFIG)
    analytics = make_elm("nvdsanalytics", "analytics")
    analytics.set_property("config-file", ANALYTICS_CONFIG)

    # ---- 追蹤器：nvdcf 模式才掛 nvtracker；BoxMOT 模式在 pgie.src 探針內處理 ----
    tracker = None
    if TRACKER_MODE == "nvdcf":
        tracker = make_elm("nvtracker", "tracker")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        tracker.set_property("ll-lib-file", resolve_tracker_lib())
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)

    pipeline_elements = [q1, preprocess, q2, pgie, q3, q_sgie_plate, sgie_plate, q_sgie_num, sgie_num, q_analytics, analytics, q4]
    if tracker: pipeline_elements.append(tracker)
    for elm in pipeline_elements: g_pipeline.add(elm)

    # ---- 串接中段（nvdcf 模式插入 tracker，否則跳過）----
    streammux.link(q1); q1.link(preprocess); preprocess.link(q2); q2.link(pgie); pgie.link(q3)
    if TRACKER_MODE == "nvdcf": q3.link(tracker); tracker.link(q_sgie_plate)
    else: q3.link(q_sgie_plate)
    q_sgie_plate.link(sgie_plate); sgie_plate.link(q_sgie_num); q_sgie_num.link(sgie_num)
    sgie_num.link(q_analytics); q_analytics.link(analytics); analytics.link(q4)

    # ---- 掛載探針 ----
    if TRACKER_MODE == "nvdcf": tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)
    else: pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, boxmot_pgie_src_probe, 0)
    sgie_plate.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, expand_plate_probe, 0)
    sgie_num.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, assemble_plate_probe, 0)

    # ---- demux：把 batch 拆回每路，分別接顯示/存檔/RTSP ----
    demux = make_elm("nvstreamdemux", "demuxer")
    g_pipeline.add(demux)
    q4.link(demux)

    display_streammux = _build_display_sink(g_pipeline, num_sources) if show_window else None
    for pad_index, cfg in SOURCE_CONFIGS.items():
        setup_cam_branch(g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe)

    # ---- 訊號處理 +（有終端機才）鍵盤監聽 + 主迴圈 ----
    g_loop = GLib.MainLoop()

    # systemctl stop/restart 送 SIGTERM；Ctrl+C 送 SIGINT。在主迴圈內安全處理：
    # 送 EOS 讓影片收尾，再排程強制退出，確保 finally 的 force_finalize_all() 一定跑到。
    def _on_stop_signal(_user_data):
        global g_eos_triggered
        print("\n[INFO] 收到停止訊號（SIGTERM/SIGINT），準備安全退出並存檔...")
        if g_pipeline and not g_eos_triggered:
            g_eos_triggered = True
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        return GLib.SOURCE_CONTINUE

    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, _on_stop_signal, None)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, _on_stop_signal, None)

    # 只有真的有終端機（互動執行）時才設鍵盤監聽。
    # systemd 服務底下沒有 TTY，這段必須跳過，否則 termios.tcgetattr 會直接報錯、程式還沒跑就掛。
    interactive = sys.stdin.isatty()
    fd = None
    old_settings = None
    if interactive:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按下 'q' 鍵即可優雅退出並存檔...\n")
    else:
        print("\n[INFO] 非互動模式（無終端機）：鍵盤監聽停用，請用 'systemctl stop' 安全停止。\n")

    try:
        bus = g_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, g_loop)

        g_pipeline.set_state(Gst.State.PLAYING)

        # 看門狗：只有存在 RTSP 路時才啟動（檔案批次跑不需要，也不該監控）
        if g_sources:
            GLib.timeout_add_seconds(WATCHDOG_CHECK_SEC, _watchdog_check)
            print(f"[INFO] 看門狗啟動：監控 {len(g_sources)} 路即時串流，"
                  f"每 {WATCHDOG_CHECK_SEC}s 檢查，卡死門檻 {WATCHDOG_STALL_SEC}s，重啟寬限 {WATCHDOG_GRACE_SEC}s")

        g_loop.run()
    finally:
        if interactive and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        force_finalize_all()
        g_pipeline.set_state(Gst.State.NULL)
        if g_obj_enc_ctx is not None:
            try:
                pyds.nvds_obj_enc_destroy_context(g_obj_enc_ctx)
            except Exception as e:
                print(f"[WARNING] 銷毀 Object Encoder context 失敗：{e}")


if __name__ == '__main__':
    initialize_state_managers()
    main()
