#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepStream 車牌辨識 (LPR) 設定檔產生器

功能說明：
    掃描指定資料夾中的影片檔（支援 .mp4, .avi, .mov, .mkv），
    根據影片檔名解析真實起始時間，並為每一支影片產生一個對應的 YAML 設定檔，
    供 DeepStream LPR 管線使用。

設定方式：
    修改下方「全域設定區」的變數即可調整所有產出的 YAML 內容。
"""

import os
import re
from pathlib import Path


# ------------------------------------------------------------
# 全域設定區
# ------------------------------------------------------------

VIDEO_DIR   = "/mnt/d/merge_video/"                # 輸入影片所在的資料夾路徑
OUTPUT_DIR  = "ds_yaml"                             # 產生的 YAML 設定檔輸出目錄
VIDEO_EXTS  = {'.mp4', '.avi', '.mov', '.mkv'}      # 支援的影片副檔名（小寫）

# ------------------------------------------------------------
# DeepStream 管線共用參數
# ------------------------------------------------------------

DEVICE_CODE = "EdgeX317"                            # 邊緣設備代碼（寫入 DB 的 DeviceCode 欄位）
STREAM_FPS  = 30.0                                  # 串流目標影格率 (FPS)

# ------------------------------------------------------------
# 畫面解析度（來源真實解析度）
# ------------------------------------------------------------
# 供 ROI/crop 座標自動換算到 1920×1080 用：若來源不是 1080p，改成來源真實解析度即可，
# ROI/crop 會依比例自動對齊（詳見 logic/config.py 的 _scale_points）。
BASE_W = 1920                                       # 來源畫面寬度
BASE_H = 1080                                       # 來源畫面高度

# ------------------------------------------------------------
# 模型 Engine 參數
# ------------------------------------------------------------
WEIGHT_IMGSZ          = 640                         # car / plate 模型的輸入解析度
NUM_IMGSZ             = 320                         # num 模型的輸入解析度
WEIGHT_BATCH_SIZE     = 4                           # car / plate engine 的最大批次量
NUM_WEIGHT_BATCH_SIZE = 16                          # num engine 的最大批次量

# ------------------------------------------------------------
# ROI 計數區域（可定義多組）
# ------------------------------------------------------------
# 格式：{ "ROI名稱": [ [x1,y1], [x2,y2], ... ] }
# 每一組 ROI 代表一個計數區域，物件每通過一組 ROI 就會寫入一筆 DB 紀錄。
# 可根據需求新增更多 ROI（例如 roi_2、roi_3…），名稱不可重複。
ROI_REGIONS = {
    "roi_1": [
        [0,    100],
        [1920, 100],
        [1920, 1080],
        [0,    1080],
    ],
    # 以下為第二組 ROI 範例（需使用時請取消註解並填入實際座標）
    # "roi_2": [
    #     [0,    540],
    #     [1920, 540],
    #     [1920, 1080],
    #     [0,    1080],
    # ],
}

# ------------------------------------------------------------
# 裁切遮罩座標（矩形以外的區域不送入 PGIE）
# ------------------------------------------------------------
# 座標必須形成封閉多邊形，通常為矩形。此區域外的畫面不會進行車輛偵測，可節省運算資源。
CROP_POINTS = [
    [0,    50],
    [1920, 50],
    [1920, 1080],
    [0,    1080],
]

# ------------------------------------------------------------
# 輸出開關
# ------------------------------------------------------------

SAVE_OUTPUT_VIDEO  = True                           # 是否儲存推理後的影片（含標記框）
SAVE_OUTPUT_DB     = True                           # 是否將計數結果寫入 SQLite 資料庫
SAVE_SCREENSHOT    = True                           # 是否啟用截圖功能（車輛框 + 車牌框）
OUTPUT_VIDEO_DIR   = "/mnt/d/output_video"          # 推理後影片的輸出目錄

# ------------------------------------------------------------
# 本地顯示開關
# ------------------------------------------------------------

SHOW_WINDOW        = True                           # 是否顯示即時預覽視窗
SHOW_FPS_OVERLAY   = True                           # 是否在畫面左上角顯示 FPS
SHOW_ROI           = True                           # 是否在畫面繪製 ROI 區域的黃色框線
SHOW_CROP          = True                           # 是否在畫面繪製裁切區域的青綠色框線

# ------------------------------------------------------------
# 追蹤器類型
# ------------------------------------------------------------
# 可選值：'nvdcf', 'bytetrack', 'ocsort', 'fasttracker', 'sfsort', 'cbiou'
TRACKER_TYPE = "cbiou"

# ------------------------------------------------------------
# 抖動過濾與容忍參數
# ------------------------------------------------------------
AXIS               = "y"                            # 進出方向判斷軸："y"（上下）或 "x"（左右）
UP_LEFT_IS_OUT     = True                           # True ：往上/往左=OUT、往下/往右=IN
                                                    # False：反轉，往上/往左=IN、往下/往右=OUT
MOVEMENT_THRESHOLD = 70                             # 所選軸的位移門檻（像素）：位移 > 此值 / < -此值 才定向，否則視為抖動
MIN_ROI_HITS       = 2                              # ROI 命中最低影格數（避免短暫經過誤判）

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 輔助函式
# ------------------------------------------------------------
def yaml_bool(value: bool) -> str:
    """
    將 Python 布林值轉換為 YAML 格式的小寫 true/false 字串。

    Args:
        value: 要轉換的布林值。

    Returns:
        若 value 為 True 回傳 "true"，否則回傳 "false"。
    """
    return "true" if value else "false"


def format_points(points: list, indent: int) -> str:
    """
    將 [[x, y], ...] 座標列表轉換為 YAML 多行列表字串。

    Args:
        points: 座標列表，每個元素為 [x, y]。
        indent: 每行前方的空格數量（縮排層級）。

    Returns:
        格式化後的 YAML 字串，例如：
              - [0, 100]
              - [1920, 100]
    """
    pad = " " * indent
    return "\n".join(f"{pad}- [{x}, {y}]" for x, y in points)


def format_regions(regions: dict) -> str:
    """
    將 ROI 區域字典轉換為 YAML 區塊字串。

    Args:
        regions: 字典，鍵為 ROI 名稱，值為座標列表。

    Returns:
        格式化後的 YAML 字串，例如：
            roi_1:
              - [0, 100]
              - [1920, 100]
            roi_2:
              ...
    """
    blocks = []
    for name, points in regions.items():
        block = f"    {name}:\n" + format_points(points, indent=6)
        blocks.append(block)
    return "\n".join(blocks)


def get_yaml_content(source_id: str, video_source_path: str, start_time_str: str) -> str:
    """
    產生完整的 DeepStream LPR YAML 設定檔內容（字串形式）。

    Args:
        source_id: 來源代號（取自影片檔名，不含副檔名）。
        video_source_path: 影片檔案實際路徑（含檔名），寫入 YAML 的 source 欄位。
        start_time_str: 影片起始時間，格式為 "YYYY-MM-DD HH:MM:SS"。

    Returns:
        符合 DeepStream LPR 格式的 YAML 字串。
    """
    regions_block = format_regions(ROI_REGIONS)            # 多組 ROI
    crop_block    = format_points(CROP_POINTS, indent=4)   # 裁切遮罩座標

    # 將 Python 布林值轉為 YAML 字串
    save_output_video = yaml_bool(SAVE_OUTPUT_VIDEO)
    save_output_db    = yaml_bool(SAVE_OUTPUT_DB)
    save_screenshot   = yaml_bool(SAVE_SCREENSHOT)
    show_window       = yaml_bool(SHOW_WINDOW)
    show_fps_overlay  = yaml_bool(SHOW_FPS_OVERLAY)
    show_roi          = yaml_bool(SHOW_ROI)
    show_crop         = yaml_bool(SHOW_CROP)
    up_left_is_out    = yaml_bool(UP_LEFT_IS_OUT)

    return f"""# ------------------------------------------------------------
# DeepStream 車牌辨識 (LPR) - {source_id} 設定
# ------------------------------------------------------------

source_id: "{source_id}"      # 來源代號（OSD 標記 / 輸出檔名前綴 / DB 欄位）

# ------------------------------------------------------------
# 裝置資訊
# ------------------------------------------------------------
device:
  code: "{DEVICE_CODE}"                    # 邊緣設備代碼，寫入 DB 的 DeviceCode 欄位

# ------------------------------------------------------------
# 輸入來源 ⭐
# ------------------------------------------------------------
# 可填值：
#   "videos/test1.avi"                # 影片（相對路徑，需置於 videos/ 資料夾）
#   "rtsp://user:pass@ip:port/path"   # RTSP 串流
source: "{video_source_path}"

stream_fps: {STREAM_FPS}                      # 串流影格率
start_time: "{start_time_str}"     # 影片第一幀對應的真實時間（RTSP 模式自動忽略）
                                      # 格式：YYYY-MM-DD HH:MM:SS

# ------------------------------------------------------------
# 模型 Engine ⭐
# ------------------------------------------------------------
weight_imgsz: {WEIGHT_IMGSZ}                     # car / plate 模型的輸入解析度
num_imgsz: {NUM_IMGSZ}                        # num 模型的輸入解析度
weight_batch_size: {WEIGHT_BATCH_SIZE}                  # car / plate engine 的最大批次量
num_weight_batch_size: {NUM_WEIGHT_BATCH_SIZE}             # num engine 的最大批次量

# ------------------------------------------------------------
# 偵測閾值
# ------------------------------------------------------------
detect:
  car_conf:   0.25                    # PGIE 車輛偵測信心值
  car_iou:    0.45                    # PGIE 車輛偵測 NMS IoU 門檻
  plate_conf: 0.25                    # SGIE 車牌偵測信心值
  plate_iou:  0.45                    # SGIE 車牌偵測 NMS IoU 門檻
  num_conf:   0.25                    # SGIE 字元偵測信心值
  num_iou:    0.45                    # SGIE 字元偵測 NMS IoU 門檻

# ------------------------------------------------------------
# ROI / 裁切遮罩 ⭐
# ------------------------------------------------------------
geometry:
  base_w: {BASE_W}                        # 來源畫面寬度
  base_h: {BASE_H}                        # 來源畫面高度

  # 計數 ROI（座標定義於上方 ROI_REGIONS，可定義多組且各別寫入一筆 DB）
  regions:
{regions_block}

  # 裁切遮罩：矩形以外的區域不送入 PGIE（座標定義於上方 CROP_POINTS）
  crop_points:
{crop_block}

# ------------------------------------------------------------
# 結算規則
# ------------------------------------------------------------
session:
  cleanup_frames: 30                  # 物件消失 N 幀後進行結算
  flush_interval_seconds: 5           # 每隔 N 秒將暫存資料寫入 DB（設 0 表示即時寫入）

# ------------------------------------------------------------
# 抖動過濾與容忍參數 ⭐
# ------------------------------------------------------------
# axis = 依哪一個座標軸判斷進出方向
#   "y"（預設）→ 看垂直位移：向下為 IN、向上為 OUT
#   "x"        → 看水平位移：向右為 IN、向左為 OUT
# movement_threshold = 所選軸的位移像素門檻
#   位移 > 此值 → IN（y:向下 / x:向右）
#   位移 < -此值 → OUT（y:向上 / x:向左）
#   絕對值 < 此值 → NA（視為抖動誤判，不寫 DB）
# min_roi_hits = ROI 命中最少影格數（避免短暫經過誤判）
track_logic:
  axis: "{AXIS}"
  movement_threshold: {MOVEMENT_THRESHOLD}
  min_roi_hits: {MIN_ROI_HITS}
  up_left_is_out: {up_left_is_out}                 # true ：往上/往左=OUT、往下/往右=IN
                                       # false：反轉，往上/往左=IN、往下/往右=OUT

# ------------------------------------------------------------
# 輸出設定 ⭐
# ------------------------------------------------------------
output:
  save_output_video: {save_output_video}             # 可選：true / false  是否輸出推理後影片 ❌
  output_video_dir: "{OUTPUT_VIDEO_DIR}"

  save_output_db: {save_output_db}                # 可選：true / false  是否寫入 SQLite DB
  output_db_dir: "output_db"          # DB 檔輸出資料夾

  save_screenshot: {save_screenshot}               # 可選：是否啟用截圖（車種框 + 車牌框）❌

# ------------------------------------------------------------
# 本地顯示
# ------------------------------------------------------------
display:
  show_window: {show_window}                   # 可選：true / false  本地預覽視窗 ❌
  show_fps_overlay: {show_fps_overlay}              # 可選：true / false  畫面左上 FPS
  show_roi: {show_roi}                      # 可選：true / false  ROI 黃線
  show_crop: {show_crop}                     # 可選：true / false  裁切框青綠線

# ------------------------------------------------------------
# 追蹤器 ⭐
# ------------------------------------------------------------
# 調整追蹤器內部參數請編輯 boxmot/configs/trackers/<type>.yaml
# 可選值：'nvdcf', 'bytetrack', 'ocsort', 'fasttracker', 'sfsort', 'cbiou'
tracker:
  type: "{TRACKER_TYPE}"                       # ⭐ 改這裡切換追蹤器
"""


def parse_timestamp(filename: str):
    """
    從影片檔名解析真實起始時間。

    支援的檔名時間格式：
        1. YYYYMMDD_HHMMSS 或 YYYYMMDD-HHMMSS
           範例：華D_20260527_015223.mp4
        2. YYYYMMDDHHMMSS（連續 14 碼數字）
           範例：xxx-20250522002357.mp4
        3. YYYY_MMDD_HHMMSS 或 YYYY-MMDD-HHMMSS
           範例：xxx-2025_1217_000000.mp4

    Args:
        filename: 影片檔案名稱（不含路徑）。

    Returns:
        成功解析時回傳 "YYYY-MM-DD HH:MM:SS" 格式的字串，
        無法解析則回傳 None。
    """
    # 格式 1：YYYYMMDD_HHMMSS 或 YYYYMMDD-HHMMSS
    match = re.search(r'(\d{8})[-_](\d{6})', filename)
    if match:
        date_str = match.group(1)   # YYYYMMDD
        time_str = match.group(2)   # HHMMSS
        return (
            f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} "
            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )

    # 格式 2：連續 14 碼數字 (YYYYMMDDHHMMSS)
    match = re.search(r'(\d{14})', filename)
    if match:
        dt = match.group(1)
        return (
            f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]} "
            f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}"
        )

    # 格式 3：YYYY_MMDD_HHMMSS 或 YYYY-MMDD-HHMMSS
    match = re.search(r'(\d{4})[-_](\d{4})[-_](\d{6})', filename)
    if match:
        year, md, time_str = match.groups()
        month = md[:2]
        day   = md[2:4]
        return (
            f"{year}-{month}-{day} "
            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )

    # 不符合任何已知格式
    return None


# ------------------------------------------------------------
# 主程式
# ------------------------------------------------------------
def main():
    """
    主程式：掃描影片資料夾、解析時間戳、產生 YAML 設定檔。
    """
    input_path = Path(VIDEO_DIR)

    if not input_path.exists():
        print(f"錯誤：找不到資料夾 '{VIDEO_DIR}'")
        return

    print(f"正在掃描資料夾：{VIDEO_DIR} ...")
    count   = 0   # 成功產生的 YAML 檔案數量
    skipped = 0   # 因無法解析時間而跳過的檔案數量

    files = sorted(input_path.iterdir())

    for file in files:
        if file.is_file() and file.suffix.lower() in VIDEO_EXTS:
            file_stem = file.stem          # 檔名（不含副檔名）→ 作為 source_id

            # 解析影片起始時間
            start_time = parse_timestamp(file.name)
            if not start_time:
                print(f"[跳過] 無法解析時間：{file.name}")
                skipped += 1
                continue

            # 組合 YAML 中的 source 路徑（使用正斜線，相容於各種作業系統）
            video_source_path = f"{VIDEO_DIR.rstrip('/')}/{file.name}"

            # 產生 YAML 內容
            yaml_content = get_yaml_content(file_stem, video_source_path, start_time)

            # 寫入輸出檔案
            output_filename = f"{file_stem}.yaml"
            output_path = Path(OUTPUT_DIR) / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            print(f"[生成] {output_filename}\t(時間：{start_time})")
            count += 1

    print(f"\n✅ 完成！共生成 {count} 個 YAML，跳過 {skipped} 個，檔案位於 '{OUTPUT_DIR}' 資料夾。")


if __name__ == "__main__":
    main()
