# DeepStream 車牌辨識 (LPR) 專案 — 功能總結

本專案基於 **DeepStream 7.1**（TensorRT 10.3）打造，採 **三層推論架構**（車輛偵測 → 車牌框偵測 → 車牌字元辨識），支援**多路影像同時辨識**，具備跨平台執行（Jetson / dGPU / WSL2 自動適配）、解析度自動換算、逐路方向判定、類別過濾、車牌組字與截圖、每路獨立資料庫等功能。以下依模組整理目前所有功能。

---

## 一、三層推論架構（核心）

- **三層 GIE 串接**：`PGIE 車輛偵測 → [tracker] → SGIE 車牌框偵測 → SGIE 車牌字元辨識 → nvdsanalytics`。三個 engine 的 `gie-unique-id` 分別為 1（車輛）、2（車牌）、3（字元）。
- **單一 pipeline、多路來源**：所有 cam 共用一條 pipeline，`streammux` 的 batch 自動等於 cam 數；demux 後再逐路做 OSD 與顯示 / 寫檔。
- **preprocess 裁切 ROI**：`nvdspreprocess` 依各路 `crop_points` 只把矩形內畫面送進 PGIE，節省算力（張量規格由 `weight_imgsz` / `weight_batch_size` 帶入）。
- **num explicit-batch（固定批次）**：字元模型用 explicit-batch（full-dims）模式，帶 `force-implicit-batch-dim=0`（nvinfer 預設，非「動態」之意）與 `infer-dims=3;H;W` 明確宣告輸入維度。engine 是靜態或動態取決於 ONNX / trtexec 如何 build；重點是設定的 `batch-size` 要對齊 num engine 實際 build 的批次，否則會被 DeepStream 蓋回 engine 真正批次。
- **設定檔自動產生**：`LPR_txt.py` 讀 YAML 產生 8 份 DeepStream 設定檔（app / preprocess / 三層 infer / analytics / tracker runtime / mux）。

## 二、車牌組字與截圖

- **車牌框外擴**：字元偵測前把車牌框四邊各外擴 10%，避免字元貼邊被裁掉。
- **字元組字**：對單台車的字元框先做 NMS 去重，再依框中心 x 由左到右排序，查表組成車牌字串。
- **截圖（依 PTS 時間戳對齊）**：影像不是從 batch buffer 直接讀，而是在 `streammux` 之前由 `tee` 分出 `appsink`，把每格 RGBA（系統記憶體）連同 PTS 存入環形緩存；各 probe 再用 `frame_meta.buf_pts` 找「PTS 最接近的那格」裁切。即使丟幀或啟動差一格，也只會少存幾張，不會裁到錯誤畫面。
- **兩種截圖**：車牌截圖（取面積最大的一次）與車種截圖 fallback（無車牌時取整車面積最大的一次），存檔路徑寫進 DB 的 `PlateImg` / `ClassImg`。

## 三、類別過濾（keep_classes）

- **逐路類別白名單**：每個 YAML 可寫 `keep_classes: [0]`，只保留指定的原始 class_id，其餘車輛偵測框在 probe 階段就丟掉（不畫框、不追蹤、不計數、不寫 DB）。不寫代表全收，行為與原本一致。
- nvdcf 與 BoxMOT 兩種追蹤模式都套用同一過濾。

## 四、解析度自動換算（Auto Resize）

- **ROI / crop 自動縮放**：在 YAML 按**來源真實解析度**標記 ROI / `crop_points`，程式依該路 `base_w/base_h` 與 streammux 輸出（固定 1920×1080）的比例自動換算成 1080P 座標。720P、4K 等任何來源都可直接寫真實點位，不用手算。
- **三處座標系一致**：probe 計數用的 `cv_regions`、preprocess 裁切框、nvdsanalytics ROI 視覺線，三者都套同一換算，不會歪掉。
- **混解析度可行**：同時跑 1080p / 720p / 4K 沒問題，streammux 統一縮放成 1080P 後再 batch。

## 五、方向判定（IN / OUT）

- **逐路獨立軸向**：每路可各自設 `track_logic.axis: y`（上下）或 `x`（左右），互不影響。
- **方向反轉開關**：`up_left_is_out`（預設 `true`）決定「往上/往左＝OUT、往下/往右＝IN」；設 `false` 整個對調，鏡頭裝反時免重畫 ROI。
- **方向值**：`IN` / `OUT`，位移不足則判為 `NA`、不寫 DB。
- **抖動過濾**：`movement_threshold`（位移門檻）、`min_roi_hits`（ROI 命中最少幀數）防止誤判。

## 六、追蹤器

- **雙模式**：`nvdcf`（DeepStream 內建 nvtracker）或 **BoxMOT** 系列（`bytetrack / ocsort / fasttracker / sfsort / cbiou`），由 YAML `tracker.type` 決定。
- BoxMOT 模式在 `pgie.src` 探針接管：抽偵測框 → 餵 BoxMOT → 用追蹤結果重建 obj_meta 供後續 SGIE 使用。
- **追蹤器 .so 路徑自動偵測**：nvtracker 的 `libnvds_nvmultiobjecttracker.so` 依環境變數 `DS_TRACKER_LIB` →常見安裝路徑→標準 `/opt/nvidia` 順序自動解析，跨版本 / 跨機器不用手改。

## 七、多 ROI 計數

- **單路多 ROI**：一台車經過多個 ROI 各寫一筆，ROI 名稱寫進 DB 的 `ROI` 欄位。
- **消失才結算**：ID 連續 `cleanup_frames` 幀沒再出現才結算寫出；結束時強制結算所有殘留軌跡。

## 八、資料庫（每路獨立、明細版）

- **每路一個 DB**：各 cam 各自寫進自己的 `output_db/<source_id>.db`（非合併單檔）。
- **每台車一筆明細**：ID 消失結算時寫一筆。
- **`events` 表欄位**：`DeviceCode / CameraCode / TrackID / Plate / Class / ROI / Direction / HitCount / VideoTime / CreateTime / ClassImg / PlateImg`。
- **CreateTime 雙模式**：檔案 = `start_time + 影片虛擬秒數`；RTSP = 系統當下時間。
- **WAL 模式**：支援邊寫邊讀，會伴隨 `-wal` / `-shm` 檔，屬正常現象。
- **save_output_db=false**：只印 log、不寫 DB。
- **local_id 循環**：每路各自 1 ～ 999999 循環（撞號靠 CreateTime 區分）。
- 匯出：`db_to_excel.py` 可把 DB 轉成 Excel（DB 格式不變）。

## 九、跨平台 / 輸入 / 輸出 / 顯示

- **編碼器自動偵測**：偵測到 NVENC（`nvv4l2h264enc`）走硬體編碼，否則自動退回 CPU 軟體編碼（x264 / x265）；可用環境變數 `USE_CPU_ENCODER=1/0` 強制覆寫。→ Jetson / dGPU 自動用硬編，WSL 無 NVENC 時自動用 CPU。
- **顯示 sink 自動退路**：優先 NVIDIA sink（`nveglglessink` / `nv3dsink`）；找不到時退回標準 GStreamer sink（`ximagesink` / `glimagesink` / `autovideosink`），可用 `DS_DISPLAY_SINK` 指定。純 WSLg / dGPU 也能顯示。
- **多來源類型**：本地影片檔、RTSP、HTTP。
- **HEADLESS 模式**：`run_batches.sh` 內 `HEADLESS=0/1` 切換；或把 YAML 的 `show_window` 全設 false，即不建顯示分支、純跑辨識與寫檔。
- **畫面疊加**：bbox 用車種色、ID + 車種標籤、車牌字串、左上角即時 FPS。
- **寫檔輸出**：可選存推論後影片（含 videorate 穩定 PTS）。

## 十、執行 / 結束

- **批次腳本 `run_batches.sh`**：含 HEADLESS 開關、總耗時統計、Ctrl+C 安全退出。
- **安全退出**：終端機按 `q` 或收到 SIGINT / SIGTERM，送 EOS，全部封裝完才退出（逾時 fallback）。
- **結束強制結算**：把還在畫面內的殘留軌跡全部結算寫出。
- **每 30 秒 FPS 報告**（涵蓋所有 cam）。

---

## 執行流程

```bash
# 1. 產生設定檔（改過 YAML 的 ROI / crop / engine / tracker 後都要重跑）
python LPR_txt.py

# 2. 啟動辨識
python main.py

# 或用批次腳本（可切 HEADLESS）
HEADLESS=1 ./run_batches.sh
```

> **提醒**：更動 ROI / `crop_points` / engine 參數後，務必先重跑 `LPR_txt.py` 再跑 `main.py`，因為裁切框與 infer 設定是在產生設定檔階段計算的。

> **顯示 / 編碼相依**：在無 NVENC 的環境（如部分 WSL）會走 CPU 編碼，需安裝 GStreamer 外掛：
> `sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav`

---

## 檔案結構與職責

| 檔案 | 職責 |
|------|------|
| `main.py` | 建立三層 pipeline、掛探針、mainloop、安全退出；tracker `.so` 路徑自動偵測 |
| `LPR_txt.py` | 讀 YAML 產生 8 份 DeepStream 設定檔（含 ROI/crop 自動縮放到 1080P、preprocess/tracker 路徑自動偵測） |
| `gen_DS_yaml.py` | 掃描影片資料夾、依檔名解析起始時間，批次產生各路 YAML（頂部變數區可調） |
| `logic/config.py` | 載入 YAML、建立全域 `SOURCE_CONFIGS`、ROI/crop 自動縮放、`keep_classes`、`axis`/`up_left_is_out`、追蹤器模式 |
| `logic/pipeline.py` | GStreamer 元件建構、編碼器自動偵測、顯示 sink 退路、每路下游分支（顯示 / 寫檔） |
| `logic/probes.py` | 追蹤探針（nvdcf / BoxMOT）、方向判定、多 ROI 命中、類別過濾、車牌組字、截圖、OSD |
| `logic/state_db.py` | 每路獨立 SQLite、每台車明細結算與寫入、截圖存檔、local_id 管理 |
| `logic/boxmot_adapter.py` | BoxMOT 追蹤器介接 |
| `logic/color.py` | 類別標籤（車種 / 字元）與顏色 |
| `db_to_excel.py` | 把 DB 匯出成 Excel |
| `ds_yaml/*.yaml` | 每路 cam 的設定（來源、engine、ROI、方向、追蹤器等） |

---

## YAML 設定範例重點

```yaml
source_id: "camC"                 # 寫進 DB CameraCode / OSD / 輸出檔名前綴
device: {code: "EdgeX317"}        # 寫進 DB DeviceCode

source: "videos/test3.mp4"        # 影片檔 / rtsp:// / http://
stream_fps: 15.0
start_time: "2025-05-22 00:23:57" # 影片首幀對應真實時刻（RTSP 自動忽略）

weight_imgsz: 640                 # car / plate 模型輸入解析度
num_imgsz: 320                    # num 模型輸入解析度
weight_batch_size: 2              # car / plate engine max batch
num_weight_batch_size: 16         # num engine max batch

detect:
  car_conf: 0.25 ;  car_iou: 0.45
  plate_conf: 0.25; plate_iou: 0.45
  num_conf: 0.25 ;  num_iou: 0.45

keep_classes: [0]                 # （選填）只保留原始 class_id=0；不寫=全收

geometry:
  base_w: 1920                    # ⭐ 來源真實寬（ROI/crop 自動縮放的依據）
  base_h: 1080                    # ⭐ 來源真實高
  regions:                        # 計數 ROI，直接用來源真實點位標記
    roi_1: [[0,100],[1920,100],[1920,1080],[0,1080]]
  crop_points: [[0,50],[1920,50],[1920,1080],[0,1080]]   # 裁切遮罩

track_logic:
  axis: "y"                       # 'y'=上下判進出 / 'x'=左右判進出
  movement_threshold: 30
  min_roi_hits: 2
  up_left_is_out: true            # true:往上/往左=OUT、往下/往右=IN；false:反轉

tracker:
  type: "cbiou"                   # nvdcf / bytetrack / ocsort / fasttracker / sfsort / cbiou
```

---

## 與「多權重車流計數版」的差異

| 面向 | 本專案（LPR） | 多權重車流計數版 |
|------|----------------|-------------------|
| 推論架構 | 三層 GIE（車→車牌→字元），單一 pipeline 多路 | 依 weight 分組，每組一條 pipeline |
| 資料庫 | 每路一個 `.db` | 單一合併 `traffic_count.db` |
| 方向值 | `IN` / `OUT` | `flow_in` / `flow_out` |
| 額外產物 | 車牌字串、車牌 / 車種截圖 | — |
| 共通功能 | 解析度自動換算、`axis` + `up_left_is_out`、`keep_classes`、跨平台編碼 / 顯示、HEADLESS、追蹤器路徑自動偵測 | 同左 |
