"""
全域設定載入器（車牌辨識 LPR 版）

主要功能：
1. 自動偵測專案根目錄 BASE_DIR，組出所有 DeepStream 設定檔的絕對路徑
2. 載入 ds_yaml/*.yaml，建立全域 SOURCE_CONFIGS 字典（每路 cam 一筆設定）
3. YAML source 智慧解析：支援相對路徑、絕對路徑、RTSP、HTTP、樣板等多種寫法
4. 載入 config_tracker_runtime.txt，決定追蹤器模式 (nvdcf / BoxMOT 系列)
5. 後處理 YAML 欄位：ROI 轉 numpy、RTSP 帳密編碼、影片 FPS 自動偵測
6. 截圖設定：依 save_screenshot 旗標決定資料夾根名（影片檔名 / source_id）並建立目錄
7. 暴露 LPR 三層 GIE 設定路徑常數給 main.py 使用：
   - INFER_CONFIG            : PGIE 車輛偵測
   - INFER_SEC_PLATE_CONFIG  : SGIE 車牌偵測
   - INFER_SEC_NUM_CONFIG    : SGIE 字元偵測
"""

import os
import sys
import glob
import cv2
import yaml
import urllib.parse
import configparser
import numpy as np


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# 自動偵測專案根目錄 (本檔位於 <project_root>/logic/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# YAML_DIR = f"{BASE_DIR}/ds_yaml"
YAML_DIR = os.environ.get("DS_YAML_DIR", f"{BASE_DIR}/ds_yaml")

# --- DeepStream 主要設定檔路徑 ---
INFER_CONFIG       = f"{BASE_DIR}/config_infer_primary_yolo11.txt"
TRACKER_CONFIG     = f"{BASE_DIR}/config_tracker_NvDCF_accuracy.yml"
PREPROCESS_CONFIG  = f"{BASE_DIR}/config_preprocess.txt"
ANALYTICS_CONFIG   = f"{BASE_DIR}/config_nvdsanalytics.txt"

# --- LPR 特有：兩層 SGIE 設定路徑 ---
INFER_SEC_PLATE_CONFIG = f"{BASE_DIR}/config_infer_secondary_plate.txt"
INFER_SEC_NUM_CONFIG   = f"{BASE_DIR}/config_infer_secondary_num.txt"

# --- 追蹤器執行期設定 (由 LPR_txt.py 產生) ---
TRACKER_RUNTIME_CONFIG = f"{BASE_DIR}/config_tracker_runtime.txt"

# --- 截圖根目錄 ---
SCREENSHOT_ROOT = f"{BASE_DIR}/screenshot"

# --- 支援的 URI scheme ---
_URI_SCHEMES = ("file://", "rtsp://", "rtsps://", "http://", "https://")

# --- Local ID 循環上限 ---
# 每路 cam 各自的 local_id 累積到此值後歸 1 重新計算（OSD 與 DB TrackID 同步循環）
# 撞號的紀錄靠 CreateTime 區分；查 DB 時記得帶時間範圍
LOCAL_ID_MAX = 999999

# --- streammux 輸出尺寸（固定 1920×1080，跨解析度來源座標對齊用）---
# 所有來源進 pipeline 後都由 streammux 縮放到這個尺寸；ROI/crop 的座標系一律以此為準。
# 使用者在 YAML 依「來源真實解析度」標點位，載入時自動依 base_w/base_h 比例換算到此尺寸，
# 如此即使同時跑不同解析度來源（1080p + 720p + 4K），所有 ROI 也都對得齊。
MUX_OUTPUT_W = 1920
MUX_OUTPUT_H = 1080


# ==========================================
# 2. YAML source 智慧解析 (Source URI Resolver)
# ==========================================

def _scale_points(points, base_w, base_h):
    """
    把「來源真實解析度(base_w×base_h)」座標系的點位換算成 streammux 輸出(1920×1080)座標系。

    參數：
        points (list): [[x,y], ...] 來源座標系點位
        base_w/base_h (int): 該來源的真實解析度（YAML geometry.base_w/base_h）
    返回：
        list: [[x,y], ...] 換算成 1920×1080 座標系後的整數點位（比例 1:1 時原樣回傳）
    """
    if not points:
        return points
    if not base_w or not base_h:
        return points
    sx = MUX_OUTPUT_W / float(base_w)
    sy = MUX_OUTPUT_H / float(base_h)
    if sx == 1.0 and sy == 1.0:
        return points
    return [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in points]

def _resolve_source_uri(raw):
    """
    把 YAML 的 source 欄位轉成 GStreamer 可用的合法 URI

    支援寫法：
        "videos/test3.mp4"               相對路徑 → 自動補 BASE_DIR 與 file:// 前綴
        "/abs/path/to/file.mp4"          絕對路徑 → 自動補 file:// 前綴
        "~/Videos/x.mp4"                 家目錄展開
        "${BASE_DIR}/videos/x.mp4"       樣板展開
        "file:///already/uri/x.mp4"      原樣使用
        "rtsp://user:pass@ip:port/path"  原樣使用
        "http://..." / "https://..."     原樣使用

    參數：
        raw (str): YAML 內原始字串

    返回：
        str: 解析後的 URI；空字串代表 raw 不合法
    """
    if not isinstance(raw, str) or not raw:
        return ""

    s = raw.strip()

    # 樣板展開
    if "${BASE_DIR}" in s:
        s = s.replace("${BASE_DIR}", BASE_DIR)

    # 已是合法 URI → 原樣回傳
    if s.startswith(_URI_SCHEMES):
        return s

    # 視為檔案路徑：展開家目錄、補絕對路徑
    if s.startswith("~"):
        s = os.path.expanduser(s)
    if not os.path.isabs(s):
        s = os.path.normpath(os.path.join(BASE_DIR, s))
    else:
        s = os.path.normpath(s)

    # 路徑容易拼錯，啟動時直接檢查存在
    if not os.path.exists(s):
        print(f"[WARNING] source 對應的檔案不存在：{s}")
        print(f"[WARNING]   原始 YAML 寫法：'{raw}'")
        print(f"[WARNING]   pipeline 啟動時很可能失敗，請確認路徑與檔名")

    return f"file://{s}"


def _parse_start_time(start_time_str):
    """
    將 YAML 字串解析成 datetime 物件

    參數：
        start_time_str (str): "YYYY-MM-DD HH:MM:SS" 格式字串

    返回：
        datetime | None: 解析成功回傳 datetime，失敗或空值回傳 None
    """
    if not start_time_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(str(start_time_str), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[WARNING] start_time 格式錯誤 ({start_time_str})，將忽略此欄位: {e}")
        return None


def _resolve_screenshot_dirname(source_uri, cam_name):
    """
    決定截圖資料夾的根名稱

    規則：
        檔案來源 → 用影片檔名去副檔名（如 忠C-20260512_145230.mp4 → 忠C-20260512_145230）
        RTSP/串流 → 用 source_id（如 camC）

    參數：
        source_uri (str): 已解析的 source URI
        cam_name (str): source_id

    返回：
        str: 資料夾根名稱
    """
    if source_uri.startswith("file://"):
        file_path = source_uri.replace("file://", "")
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        return name if name else cam_name
    # RTSP / HTTP 串流用 source_id
    return cam_name


# ==========================================
# 3. YAML 載入與後處理 (YAML Loader)
# ==========================================

def load_dynamic_configs(yaml_dir):
    """
    讀取 ds_yaml/*.yaml，逐檔解析並補上衍生欄位

    處理流程：
    1. 掃描所有 .yaml 檔，依檔名排序（決定 pad_index 0..N-1）
    2. 每個 YAML 補上衍生欄位：
       - device_code              : 寫進 DB DeviceCode 欄位
       - cv_regions               : 多 ROI 多邊形轉成 {roi_name: numpy int32 array}
       - track_logic              : movement_threshold / min_roi_hits（給 probes.py 用）
       - save_output_db           : DB 寫入旗標（false 時純跑統計）
       - db_path / excel_path     : DB 檔絕對路徑
       - video_path               : 輸出影片絕對路徑
       - save_screenshot          : 截圖旗標
       - screenshot_dir_class     : 車種截圖資料夾絕對路徑
       - screenshot_dir_lpr       : 車牌截圖資料夾絕對路徑
       - source / stream_fps      : 解析後的 URI 與真實 FPS
       - is_file_source           : 來源是否為本地檔案
       - start_time_dt            : 影片首幀對應的真實時刻
       - keep_classes             : 類別過濾白名單（frozenset 或 None）

    參數：
        yaml_dir (str): YAML 資料夾路徑

    返回：
        dict: {pad_index: cfg_dict, ...}
    """
    files = sorted(glob.glob(f"{yaml_dir}/*.yaml"))
    if not files:
        print(f"[ERROR] 找不到任何 YAML 檔於 {yaml_dir}")
        sys.exit(1)

    configs = {}
    for pad_index, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cam_name = data.get("source_id", f"cam_{pad_index}")

        # 步驟 1: 裝置代碼
        device_cfg = data.get("device", {}) or {}
        data["device_code"] = str(device_cfg.get("code", "UNKNOWN"))

        # 步驟 2: ROI / crop 依 base_w/base_h 自動換算成 1920×1080 座標系
        # 使用者在 YAML 按「來源真實解析度」標點位，這裡依比例縮放，與 streammux 縮放後的
        # 物件座標對齊（1080P 來源比例 1:1，數字不變；720P/4K 來源自動對齊）。
        geo = data.get("geometry", {}) or {}
        base_w = int(geo.get("base_w", MUX_OUTPUT_W))
        base_h = int(geo.get("base_h", MUX_OUTPUT_H))

        regions_src = geo.get("regions", {}) or {}
        regions_scaled = {}
        for roi_name, pts in regions_src.items():
            regions_scaled[roi_name] = _scale_points(pts, base_w, base_h) if pts else pts

        crop_src = geo.get("crop_points")
        crop_scaled = _scale_points(crop_src, base_w, base_h) if crop_src else crop_src

        if (base_w, base_h) != (MUX_OUTPUT_W, MUX_OUTPUT_H):
            print(f"[INFO] {cam_name} 來源座標 {base_w}x{base_h} → 自動換算成 "
                  f"{MUX_OUTPUT_W}x{MUX_OUTPUT_H}（ROI/crop 依比例縮放）")

        # 回寫縮放後的 geometry，並把 base_w/base_h 標準化為輸出尺寸
        geo["regions"] = regions_scaled
        if crop_src:
            geo["crop_points"] = crop_scaled
        geo["base_w"] = MUX_OUTPUT_W
        geo["base_h"] = MUX_OUTPUT_H
        data["geometry"] = geo

        # 多 ROI 轉 numpy（此時 regions 已是 1080P 座標，給 probe 的 pointPolygonTest 用）
        cv_regions = {}
        for roi_name, pts in regions_scaled.items():
            if pts and len(pts) >= 3:
                cv_regions[str(roi_name)] = np.array(pts, np.int32)
            else:
                print(f"[WARNING] {cam_name} 的 ROI '{roi_name}' 點數不足（{len(pts) if pts else 0}），略過")
        data["cv_regions"] = cv_regions
        if not cv_regions:
            print(f"[WARNING] {cam_name} 沒有任何有效的 ROI，將不會產生任何 DB 紀錄")

        # 步驟 3: track_logic 標準化（給 probes.py 用）
        # axis               = 依哪個座標軸判斷進出方向
        #                        "y"（預設）→ 看垂直位移：向下 / 向上
        #                        "x"        → 看水平位移：向右 / 向左
        # up_left_is_out     = 方向對應是否「往上/往左 = OUT」（預設 True）
        #                        True ：往上/往左=OUT、往下/往右=IN（與原本行為一致）
        #                        False：反轉，往上/往左=IN、往下/往右=OUT（鏡頭反向時用）
        # movement_threshold = 所選軸的位移像素門檻（絕對值小於此值視為抖動 → NA，不寫 DB）
        # min_roi_hits       = ROI 命中最少幀數（小於此值視為短暫飄過，不寫 DB）
        tl_cfg = data.get("track_logic", {}) or {}
        axis = str(tl_cfg.get("axis", "y")).lower().strip()
        if axis not in ("x", "y"):
            print(f"[WARNING] {cam_name} track_logic.axis 值非法（'{axis}'），退回 'y'")
            axis = "y"
        data["track_logic"] = {
            "axis":               axis,
            "movement_threshold": int(tl_cfg.get("movement_threshold", 30)),
            "min_roi_hits":       int(tl_cfg.get("min_roi_hits", 2)),
            "up_left_is_out":     bool(tl_cfg.get("up_left_is_out", True)),
        }

        # keep_classes：這一路只保留哪些「模型原始 class_id」（類別過濾白名單）
        #   有寫非空清單 → 標準化成 frozenset，probe 只處理清單內的 class_id，其餘偵測框丟掉
        #   沒寫 / 空清單   → None，代表全收（不過濾，維持原本行為）
        keep_raw = data.get("keep_classes")
        if keep_raw:
            data["keep_classes"] = frozenset(int(c) for c in keep_raw)
        else:
            data["keep_classes"] = None

        # 步驟 4: 輸出設定 - DB
        # 新欄位 output_db_dir / save_output_db；舊欄位 output_excel_dir 仍向下相容
        output_cfg = data.get("output", {}) or {}

        db_dir = output_cfg.get("output_db_dir")
        if not db_dir:
            db_dir = output_cfg.get("output_excel_dir", "output_db")
            if "output_excel_dir" in output_cfg:
                print(f"[INFO] {cam_name} 使用舊欄位 output_excel_dir，建議改為 output_db_dir")
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(BASE_DIR, db_dir)

        save_db = bool(output_cfg.get("save_output_db", True))
        if save_db:
            os.makedirs(db_dir, exist_ok=True)

        data["save_output_db"] = save_db
        data["db_path"]    = os.path.join(db_dir, f"{cam_name}.db")
        # excel_path 鍵保留指向 .db，state_db.py 仍用此鍵推算路徑，向下相容
        data["excel_path"] = data["db_path"]

        # 步驟 5: 輸出設定 - 影片
        video_dir = output_cfg.get("output_video_dir", "output_video")
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(BASE_DIR, video_dir)
        if output_cfg.get("save_output_video", False):
            os.makedirs(video_dir, exist_ok=True)
        data["video_path"] = os.path.join(video_dir, f"{cam_name}_output.mp4")

        # 步驟 6: 來源 URI 解析
        source_uri = _resolve_source_uri(data.get("source", ""))
        yaml_fps = data.get("stream_fps", 30.0)

        original = data.get("source", "")
        if original != source_uri:
            print(f"[INFO] {cam_name} source 解析: '{original}' → '{source_uri}'")

        # RTSP 帳密特殊字元安全編碼
        if source_uri.startswith("rtsp://"):
            try:
                parsed = urllib.parse.urlparse(source_uri)
                if parsed.username and parsed.password:
                    safe_username = urllib.parse.quote(urllib.parse.unquote(parsed.username))
                    safe_password = urllib.parse.quote(urllib.parse.unquote(parsed.password))
                    safe_netloc = f"{safe_username}:{safe_password}@{parsed.hostname}"
                    if parsed.port:
                        safe_netloc += f":{parsed.port}"
                    source_uri = urllib.parse.urlunparse((parsed.scheme, safe_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
                    print(f"[INFO] RTSP URI 安全格式: {source_uri}")
            except Exception as e:
                print(f"[WARNING] 解析 RTSP URI 失敗: {e}")

        # 檔案模式：用 cv2 抓真實 FPS 覆寫 YAML 值
        if source_uri.startswith("file://"):
            file_path = source_uri.replace("file://", "")
            cap = cv2.VideoCapture(file_path)
            try:
                if cap.isOpened():
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    if real_fps > 0:
                        yaml_fps = real_fps
            finally:
                cap.release()

        data["stream_fps"] = yaml_fps
        data["source"] = source_uri
        data["is_file_source"] = source_uri.startswith("file://")

        # 步驟 7: start_time 解析（僅檔案模式有效）
        data["start_time_dt"] = _parse_start_time(
            data.get("start_time") if data["is_file_source"] else None
        )

        # 步驟 8: 截圖設定
        # save_screenshot=true 時，依來源型態決定資料夾根名並建立 class/ 與 LPR/ 子目錄
        save_ss = bool(output_cfg.get("save_screenshot", False))
        data["save_screenshot"] = save_ss

        ss_root_name = _resolve_screenshot_dirname(source_uri, cam_name)
        ss_base = os.path.join(SCREENSHOT_ROOT, ss_root_name)
        data["screenshot_dir_class"] = os.path.join(ss_base, "class")
        data["screenshot_dir_lpr"]   = os.path.join(ss_base, "LPR")

        if save_ss:
            os.makedirs(data["screenshot_dir_class"], exist_ok=True)
            os.makedirs(data["screenshot_dir_lpr"], exist_ok=True)
            print(f"[INFO] {cam_name} 截圖已啟用，根目錄：{ss_base}")

        configs[pad_index] = data

    return configs


# ==========================================
# 4. 追蹤器執行期設定 (Tracker Runtime Config)
# ==========================================

def load_tracker_runtime():
    """
    讀 config_tracker_runtime.txt 取得當前追蹤器模式

    檔案由 LPR_txt.py 產生，內容範例：
        [tracker]
        mode=bytetrack
        config=/abs/path/to/boxmot/configs/trackers/bytetrack.yaml

    返回：
        tuple: (mode, boxmot_config_path)
            mode (str): "nvdcf" 或 BoxMOT 追蹤器名稱
            boxmot_config_path (str | None): BoxMOT 模式時為設定檔絕對路徑，nvdcf 時為 None
    """
    if not os.path.exists(TRACKER_RUNTIME_CONFIG):
        print(f"[INFO] 找不到 {TRACKER_RUNTIME_CONFIG}，預設使用 nvdcf 模式")
        return "nvdcf", None

    parser = configparser.ConfigParser()
    try:
        parser.read(TRACKER_RUNTIME_CONFIG, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] 解析 {TRACKER_RUNTIME_CONFIG} 失敗：{e}，退回 nvdcf")
        return "nvdcf", None

    if not parser.has_section("tracker"):
        print(f"[WARNING] {TRACKER_RUNTIME_CONFIG} 缺 [tracker] 區塊，退回 nvdcf")
        return "nvdcf", None

    mode = parser.get("tracker", "mode", fallback="nvdcf").lower().strip()
    boxmot_cfg = parser.get("tracker", "config", fallback=None)

    if boxmot_cfg is not None:
        boxmot_cfg = boxmot_cfg.strip() or None

    return mode, boxmot_cfg


# ==========================================
# 5. 模組初始化 (Module Initialization)
# ==========================================

# 模組載入時自動執行：印路徑、載入 YAML、載入追蹤器模式
print(f"[INFO] [config.py] BASE_DIR 自動偵測 = {BASE_DIR}")

SOURCE_CONFIGS = load_dynamic_configs(YAML_DIR)

TRACKER_MODE, BOXMOT_TRACKER_CONFIG = load_tracker_runtime()
print(f"[INFO] 追蹤器模式：{TRACKER_MODE}")
if TRACKER_MODE != "nvdcf":
    print(f"[INFO] BoxMOT 追蹤器設定來源：{BOXMOT_TRACKER_CONFIG}")
