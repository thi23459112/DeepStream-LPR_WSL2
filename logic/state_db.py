"""
================================================================================
 state_db.py — 追蹤狀態管理 + 事件結算 + SQLite 寫入 + 截圖存檔
================================================================================
本模組負責「資料層」：

  - 全域狀態字典（track_history / pending_records / fps_streams 等）由 probes 寫入。
  - 每路 cam 一個 SQLite DB（events 表），記錄每次車輛進出 ROI 的事件。
  - _finalize_one：當一台車「離場」（連續多幀消失）或程式結束強制結算時，
      彙整其車種（多數決）、車牌（多數決）、方向、各 ROI 命中次數，
      並把車種/車牌截圖寫檔，最後把事件排入 pending 等待批次寫入 DB。
  - flush_pending_to_db：把暫存事件批次寫進 SQLite（WAL 模式、交易包起來）。
  - force_finalize_all：程式結束時對所有仍在追蹤的車輛強制結算並關閉 DB。
================================================================================
"""
import os
import re
import time
import sqlite3
import threading
from collections import Counter
from datetime import timedelta

from logic.config import SOURCE_CONFIGS, LOCAL_ID_MAX, BASE_DIR
from logic.color import CLASS_MAP

# ---- 全域狀態（由 probes 模組讀寫；此處為單一真實來源）----
track_history    = {}   # (pad_index, obj_id) -> 該車追蹤狀態（票數/方向/ROI/截圖等）
pending_records  = {}   # pad_index -> 待寫入 DB 的事件 list
last_flush_times = {}   # pad_index -> 上次 flush 的時間
fps_streams      = {}   # pad_index -> {"current_fps": float, "timestamps": deque}
local_id_maps    = {}   # pad_index -> {全域 obj_id: 顯示用 local_id}
next_local_ids   = {}   # pad_index -> 下一個要發放的 local_id

_db_conns = {}              # pad_index -> sqlite3.Connection
_db_lock  = threading.Lock()  # 保護 DB 寫入（多執行緒）

# ---- events 資料表結構 ----
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceCode  TEXT    NOT NULL,
    CameraCode  TEXT    NOT NULL,
    TrackID     INTEGER NOT NULL,
    Plate       TEXT,
    Class       TEXT,
    ROI         TEXT    NOT NULL,
    Direction   TEXT    NOT NULL,
    HitCount    INTEGER NOT NULL,
    VideoTime   TEXT,
    CreateTime  TEXT    NOT NULL,
    ClassImg    TEXT,
    PlateImg    TEXT
);
CREATE INDEX IF NOT EXISTS idx_camera_time ON events (CameraCode, CreateTime);
CREATE INDEX IF NOT EXISTS idx_roi ON events (ROI);
CREATE INDEX IF NOT EXISTS idx_direction ON events (Direction);
CREATE INDEX IF NOT EXISTS idx_plate ON events (Plate);
"""


def _get_db_path(cfg, pad_index):
    """由設定推導該路 DB 檔路徑（把 excel_path 副檔名換成 .db）。"""
    excel_path = cfg.get("excel_path", f"output_db/cam_{pad_index}.db")
    base, _ = os.path.splitext(excel_path)
    return f"{base}.db"


def _open_db(pad_index, cfg):
    """開啟（或建立）該路 SQLite DB，啟用 WAL 模式並建表。"""
    db_path = _get_db_path(cfg, pad_index)
    db_dir = os.path.dirname(db_path)
    if db_dir: os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")       # 並發讀寫較友善
    conn.execute("PRAGMA synchronous=NORMAL")     # 兼顧效能與安全
    conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] SQLite DB 開啟: {db_path}")
    return conn


def _format_video_time(vsec):
    """把影片秒數格式化為 HH:MM:SS（負值/None 視為 0）。"""
    if vsec is None or vsec < 0: return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


def _sanitize_for_filename(text):
    """把字串清成可當檔名（只留英數、減號、底線），空值回 'NA'。"""
    if not text: return "NA"
    return re.sub(r"[^0-9A-Za-z\-_]", "_", str(text))


def _save_jpg_bytes(jpg_bytes, dir_path, filename):
    """
    把 JPEG bytes 寫入檔案，回傳「相對 BASE_DIR 的路徑」供存入 DB。

    參數：
        jpg_bytes (bytes): 影像位元組（None/空則不寫）
        dir_path (str):    目標資料夾
        filename (str):    檔名

    返回：
        str | None：相對路徑；失敗回 None。
    """
    if not jpg_bytes: return None
    try:
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
        with open(full_path, "wb") as f: f.write(jpg_bytes)
        rel_path = os.path.relpath(full_path, BASE_DIR)
        return rel_path
    except Exception as e:
        print(f"[WARNING] 截圖寫檔失敗 ({filename}): {e}")
        return None


def initialize_state_managers():
    """程式啟動時初始化各路的狀態容器，並（若啟用）開啟各路 DB 連線。"""
    for pad_index, cfg in SOURCE_CONFIGS.items():
        pending_records[pad_index] = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index] = {"current_fps": 0.0}
        local_id_maps[pad_index] = {}
        next_local_ids[pad_index] = 1
        if cfg.get("save_output_db", True):
            _db_conns[pad_index] = _open_db(pad_index, cfg)
        else:
            cam_name = cfg.get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} save_output_db=false，停用 DB 寫入（純跑統計）")


def get_local_id(pad_index, global_id):
    """
    把追蹤器的全域 obj_id 映射成「該路內遞增、可循環」的顯示用 local_id，
    讓畫面/報表上的 ID 較短、好讀（達 LOCAL_ID_MAX 後從 1 重新開始）。
    """
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]
        if next_local_ids[pad_index] >= LOCAL_ID_MAX:
            next_local_ids[pad_index] = 1
        else:
            next_local_ids[pad_index] += 1
    return local_id_maps[pad_index][global_id]


def _finalize_one(m_key, state, force=False):
    """
    結算單一車輛（離場或強制結算）：
      1. 方向為 NA（沒判定出進出）→ 不記錄。
      2. 沒有任何 ROI 命中達門檻（min_roi_hits）→ 不記錄。
      3. 車種/車牌取多數決，計算影片時間軸與實際時間點。
      4. 截圖寫檔：車種用 best_class_jpg（無則 fallback_class_jpg）；
         車牌用 best_plate_jpg（且需有有效車牌字串）。
      5. 對每個達標 ROI 各排一筆事件進 pending_records，等待批次寫入 DB。

    參數：
        m_key (tuple): (pad_index, obj_id)
        state (dict):  該車的追蹤狀態
        force (bool):  是否為程式結束時的強制結算（影響輸出標籤）
    """
    pad_index, obj_id = m_key
    cfg = SOURCE_CONFIGS.get(pad_index, {})
    cam_name = cfg.get("source_id", f"cam_{pad_index}")
    min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 2)

    # 1. 沒判定出方向 → 視為雜訊，不記錄
    if state.get("direction", "NA") == "NA": return

    # 2. 只保留命中次數達門檻的 ROI；全都不達標 → 不記錄
    triggered_rois = {
        roi_name: hits
        for roi_name, hits in state.get("roi_hits", {}).items()
        if hits >= min_hits
    }
    if not triggered_rois: return

    local_id = get_local_id(pad_index, obj_id)
    device_code = cfg.get("device_code", "UNKNOWN")
    direction = state["direction"]

    # 3a. 車種多數決
    if state.get("class_votes"):
        best_class_id = state["class_votes"].most_common(1)[0][0]
        cls_name = CLASS_MAP.get(best_class_id, f"Class_{best_class_id}")
    else:
        cls_name = "Unknown"

    # 3b. 車牌多數決
    plate_votes = state.get("plate_votes", Counter())
    if plate_votes: best_plate = plate_votes.most_common(1)[0][0]
    else: best_plate = "N/A"

    # 3c. 影片時間軸（以最後出現的影格數推算）與實際時間點（以來源起始時間推算）
    vsec = state["last_frame_num"] / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 4. 截圖寫檔
    class_img_rel = None
    plate_img_rel = None
    if cfg.get("save_screenshot", False):
        ts_for_name = create_time_str.replace("-", "").replace(":", "").replace(" ", "_")
        # 車種：優先用組牌時抓到的最佳整車圖，否則用追蹤期間面積最大的 fallback
        class_jpg = state.get("best_class_jpg") or state.get("fallback_class_jpg")
        if class_jpg:
            class_sub_dir = os.path.join(cfg.get("screenshot_dir_class", ""), cls_name)
            fname = f"{local_id}_{ts_for_name}.jpg"
            class_img_rel = _save_jpg_bytes(class_jpg, class_sub_dir, fname)

        # 車牌：需有有效車牌字串才存
        plate_jpg = state.get("best_plate_jpg")
        if plate_jpg and best_plate != "N/A":
            plate_safe = _sanitize_for_filename(best_plate)
            fname = f"{local_id}_{plate_safe}_{ts_for_name}.jpg"
            plate_img_rel = _save_jpg_bytes(plate_jpg, cfg.get("screenshot_dir_lpr", ""), fname)

    # 5. 每個達標 ROI 各排一筆事件
    tag = "[結算-強制]" if force else " "
    for roi_name, hit_count in triggered_rois.items():
        if not cfg.get("save_output_db", True):
            # DB 停用：只印出統計，不寫檔
            print(f"{tag}[{cam_name}] ID={local_id}, 車號={best_plate}, 車種={cls_name}, "
                  f"ROI={roi_name}, 方向={direction}, 次數={hit_count}, "
                  f"時間軸={time_axis}, 時間點={create_time_str}  (DB 已停用)")
            continue

        pending_records[pad_index].append((
            device_code, cam_name, local_id, best_plate, cls_name, roi_name, direction,
            hit_count, time_axis, create_time_str, class_img_rel, plate_img_rel,
        ))
        print(f"{tag}[{cam_name}] ID={local_id}, 車號={best_plate}, 車種={cls_name}, "
              f"ROI={roi_name}, 方向={direction}, 次數={hit_count}, "
              f"時間軸={time_axis}, 時間點={create_time_str}")


def flush_pending_to_db(pad_index):
    """
    把該路暫存的事件批次寫入 SQLite（單一交易、加鎖）。

    返回：
        int：實際寫入的筆數（無資料或無連線回 0）。
    """
    records = pending_records.get(pad_index, [])
    if not records: return 0
    conn = _db_conns.get(pad_index)
    if conn is None:
        records.clear()
        return 0
    with _db_lock:
        try:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO events "
                "(DeviceCode, CameraCode, TrackID, Plate, Class, ROI, Direction, HitCount, VideoTime, CreateTime, ClassImg, PlateImg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.execute("COMMIT")
            n = len(records)
            records.clear()
            return n
        except sqlite3.Error as e:
            try: conn.execute("ROLLBACK")
            except Exception: pass
            print(f"[ERROR] SQLite 寫入失敗 (pad_index={pad_index}): {e}")
            return 0


def force_finalize_all():
    """
    程式結束時的收尾：
      1. 對所有仍在追蹤的車輛強制結算。
      2. 把各路剩餘的暫存事件 flush 進 DB。
      3. 關閉所有 DB 連線並清空狀態。
    """
    print("\n[INFO] 開始執行強制結算...")
    for m_key, state in list(track_history.items()):
        _finalize_one(m_key, state, force=True)

    for pad_index, cfg in SOURCE_CONFIGS.items():
        n = flush_pending_to_db(pad_index)
        if n > 0:
            db_path = _get_db_path(cfg, pad_index)
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {n} 筆剩餘資料到 {db_path}")

    for pad_index, conn in list(_db_conns.items()):
        try: conn.close()
        except Exception as e: print(f"[WARNING] 關閉 DB 連線失敗 (pad_index={pad_index}): {e}")
    _db_conns.clear()
    track_history.clear()