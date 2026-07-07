# DeepStream 車牌辨識 (LPR) 專案 — 功能總結

基於 **DeepStream 7.1**（TensorRT 10.3），採三層推論（車輛 → 車牌 → 字元），支援多路同時辨識，跨平台自動適配（Jetson / dGPU / WSL2）。以下依模組列出功能。

---

## 一、三層推論架構（核心）

* 三層 GIE：`PGIE 車輛 → [tracker] → SGIE 車牌 → SGIE 字元 → nvdsanalytics`；`gie-unique-id` 分別為 1 / 2 / 3。
* 單一 pipeline、多路來源：streammux batch = cam 數；demux 後逐路做 OSD / 顯示 / 寫檔。
* preprocess 裁切：`nvdspreprocess` 依 `crop_points` 只把 ROI 內畫面送進 PGIE。
* 設定檔自動產生：`LPR_txt.py` 讀 YAML 產出 8 份 DeepStream 設定檔。

## 二、車牌組字與截圖

* 車牌框外擴 10%，避免字元貼邊被裁掉。
* 字元組字：NMS 去重 → 依框中心 x 由左到右排序 → 查表組成車牌字串。
* 截圖依 PTS 對齊：streammux 前用 `tee` 分 `appsink` 存 RGBA + PTS，probe 用 `buf_pts` 取最接近那格再裁切（丟幀也不會裁錯）。
* 兩種截圖：車牌（取面積最大一次）、車種 fallback（無車牌時取整車），路徑寫進 DB `PlateImg` / `ClassImg`。

## 三、解析度自動換算（Auto Resize）

* YAML 按「來源真實解析度」標 ROI / `crop_points`，程式依 `base_w/base_h` 自動換算成 1920×1080。
* 計數 ROI、preprocess 裁切框、analytics 視覺線三處座標系一致。
* 混解析度可行：1080p / 720p / 4K 同時跑，streammux 統一縮放成 1080P。

## 四、方向判定（IN / OUT）

* 逐路軸向：`track_logic.axis` 設 `y`（上下）或 `x`（左右）。
* 反轉開關：`up_left_is_out`（預設 `true`）；設 `false` 整個對調，鏡頭裝反免重畫 ROI。
* 方向值：`IN` / `OUT`，位移不足判 `NA`、不寫 DB。
* 抖動過濾：`movement_threshold`（位移門檻）、`min_roi_hits`（ROI 命中最少幀數）。

## 五、追蹤器

* 雙模式：`nvdcf`（內建）或 BoxMOT（`bytetrack / ocsort / fasttracker / sfsort / cbiou`），由 `tracker.type` 決定。
* BoxMOT 在 `pgie.src` 接管：抽偵測框 → 餵追蹤器 → 重建 obj_meta 供 SGIE 使用。
* 追蹤器 `.so` 路徑自動偵測：`DS_TRACKER_LIB` → 常見路徑 → `/opt/nvidia` 保底。

## 六、多 ROI 計數

* 單路多 ROI：一台車經過多個 ROI 各寫一筆，ROI 名稱寫進 DB `ROI` 欄。
* 消失才結算：ID 連續 `cleanup_frames` 幀未再出現才寫出；結束時強制結算殘留軌跡。

## 七、資料庫（每路獨立、明細版）

* 每路一個 DB：各 cam 各自寫進自己的 `output_db/<source_id>.db`（非合併單檔）。
* 每台車一筆明細：ID 消失結算時寫一筆。
* `events` 表欄位：`DeviceCode / CameraCode / TrackID / Plate / Class / ROI / Direction / HitCount / VideoTime / CreateTime / ClassImg / PlateImg`。
* CreateTime 雙模式：檔案 = `start_time + 影片虛擬秒數`；RTSP = 系統當下時間。
* WAL 模式：支援邊寫邊讀，會伴隨 `-wal` / `-shm` 檔，屬正常現象。
* `save_output_db=false`：只印 log、不寫 DB。
* local_id 循環：每路各自 1 ～ 999999 循環（撞號靠 CreateTime 區分）。
* 匯出：`db_to_excel.py` 可把 DB 轉成 Excel（可用 `OUTPUT_DIRECTIONS` 依方向正面表列輸出；DB 格式不變）。

## 八、跨平台 / 輸入 / 輸出 / 顯示

* 編碼器自動偵測：有 NVENC 走硬體編碼，否則退回 CPU（x264 / x265）；`USE_CPU_ENCODER=1/0` 可覆寫。
* 顯示 sink 退路：NVIDIA sink 找不到時退回 `ximagesink` / `glimagesink` / `autovideosink`，可用 `DS_DISPLAY_SINK` 指定。
* 來源類型：本地影片檔、RTSP、HTTP。
* HEADLESS：`run_batches.sh` 的 `HEADLESS=0/1`，或把 YAML `show_window` 全設 false。
* OSD 疊加：bbox 車種色、ID + 車種、車牌字串、左上角即時 FPS。
* num 模型：explicit-batch 固定批次（`force-implicit-batch-dim=0` + `infer-dims`）；`batch-size` 須對齊 engine 實際 build 的批次。

## 九、執行 / 結束

* `run_batches.sh`：HEADLESS 開關、總耗時統計、Ctrl+C 安全退出。
* 安全退出：`q` 或 SIGINT / SIGTERM → 送 EOS，全部封裝完才退（逾時 fallback）。
* 結束強制結算殘留軌跡；每 30 秒 FPS 報告。

---

## 執行流程

```bash
python LPR_txt.py            # 改過 ROI / crop / engine / tracker 後都要先重跑
python main.py               # 啟動辨識
# 或
HEADLESS=1 ./run_batches.sh  # 批次（可切 HEADLESS）
```

> 無 NVENC 環境（如部分 WSL）走 CPU 編碼，需先裝 GStreamer 外掛：
> `sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav`

## 檔案結構

| 檔案 | 職責 |
|------|------|
| `main.py` | 建三層 pipeline、掛探針、mainloop、安全退出；tracker `.so` 路徑自動偵測 |
| `LPR_txt.py` | 讀 YAML 產 8 份設定檔（ROI/crop 縮放到 1080P、路徑自動偵測） |
| `gen_DS_yaml.py` | 掃影片、依檔名解析起始時間，批次產生各路 YAML（頂部變數區可調） |
| `logic/config.py` | 載入 YAML、`SOURCE_CONFIGS`、ROI/crop 縮放、`axis` / `up_left_is_out`、追蹤器模式 |
| `logic/pipeline.py` | 元件建構、編碼器自動偵測、顯示 sink 退路、每路下游分支 |
| `logic/probes.py` | 追蹤探針、方向判定、多 ROI 命中、車牌組字、截圖、OSD |
| `logic/state_db.py` | 每路獨立 SQLite、每台車明細結算、截圖存檔、local_id 管理 |
| `logic/boxmot_adapter.py` | BoxMOT 追蹤器介接 |
| `db_to_excel.py` | DB 匯出 Excel（方向正面表列） |
| `ds_yaml/*.yaml` | 每路 cam 設定 |

## YAML 重點欄位

```yaml
source_id: "camC"                 # DB CameraCode / OSD / 輸出檔名前綴
source: "videos/test3.mp4"        # 影片檔 / rtsp:// / http://
stream_fps: 15.0
start_time: "2025-05-22 00:23:57" # 影片首幀真實時刻（RTSP 自動忽略）

geometry:
  base_w: 1920                    # ⭐ 來源真實解析度（ROI/crop 縮放依據）
  base_h: 1080
  regions: {roi_1: [[0,100],[1920,100],[1920,1080],[0,1080]]}   # 計數 ROI
  crop_points: [[0,50],[1920,50],[1920,1080],[0,1080]]          # 裁切遮罩

track_logic:
  axis: "y"                       # y=上下 / x=左右
  movement_threshold: 30
  min_roi_hits: 2
  up_left_is_out: true            # false=方向對調

tracker: {type: "cbiou"}          # nvdcf / bytetrack / ocsort / fasttracker / sfsort / cbiou
```

## 與多權重車流計數版差異

| 面向 | 本專案（LPR） | 多權重車流計數版 |
|------|----------------|-------------------|
| 架構 | 三層 GIE，單一 pipeline 多路 | 依 weight 分組，每組一條 pipeline |
| 資料庫 | 每路一個 `.db` | 單一合併 `.db` |
| 方向值 | `IN` / `OUT` | `flow_in` / `flow_out` |
| 額外產物 | 車牌字串、車牌 / 車種截圖 | — |
| 共通 | 解析度自動換算、`axis` + `up_left_is_out`、跨平台編碼 / 顯示、HEADLESS、追蹤器路徑自動偵測 | 同左 |
