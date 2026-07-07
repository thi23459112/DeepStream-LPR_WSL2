"""
================================================================================
 pipeline.py — GStreamer/DeepStream 元件建構與每路下游分支組裝
================================================================================
本模組集中放「建立元件、組裝 pipeline 下游」的工具函式：

  - make_elm / _safe_set：建立元件、安全設定屬性（屬性不存在不報錯）。
  - _is_jetson：判斷執行平台（Jetson 與 dGPU/WSL 在編碼器、OSD、顯示上有差異）。
  - resolve_tracker_lib：跨平台自動解析 nvtracker 的 ll-lib-file 路徑。
  - 編碼器/存檔分支：_make_encoder / _build_save_branch_*。
  - _build_display_sink：建立顯示用的第二顆 nvstreammux + tiler + sink（多路拼接顯示）。
  - setup_cam_branch：demux 之後，為「每一路」組裝 OSD 與顯示/存檔下游。

平台差異重點：
  USE_CPU_ENCODER 預設「優先 NVENC」：預設走硬體編碼，只有確實建不出 NVENC 才退回 x264/x265 CPU 編碼；
  可用環境變數 USE_CPU_ENCODER 強制覆寫（1=CPU、0=NVENC）。nvosd 的 process-mode 與顯示 sink 也依平台自動選擇。
  顯示 sink 找不到 NVIDIA sink（nveglglessink/nv3dsink）時，會退回標準 GStreamer sink
  （ximagesink/glimagesink/autovideosink，可用 DS_DISPLAY_SINK 指定）。
================================================================================
"""
import sys
from gi.repository import Gst
import os


def _detect_cpu_encoder():
    """
    決定是否使用 CPU 軟體編碼器（x264/x265）。

    規則（可用環境變數 USE_CPU_ENCODER 覆寫，1/true=強制 CPU，0/false=強制 NVENC）：
      - 環境變數有明確指定 → 依指定
      - 否則：預設優先使用 NVENC 硬體編碼（較快）；只有在「確實建不出 NVENC」時才退回 CPU，
              避免在無 NVENC 的環境（如部分 WSL）直接中斷。
    """
    env = os.environ.get("USE_CPU_ENCODER")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    # 預設 = NVENC。用「實際建立元件」測試（比 ElementFactory.find 更可靠）
    test = Gst.ElementFactory.make("nvv4l2h264enc", None)
    if test is not None:
        print("[INFO] 預設使用 NVENC 硬體編碼（nvv4l2h264enc）")
        return False
    print("[INFO] 建不出 NVENC，退回 CPU 軟體編碼（x264/x265）")
    return True


# 編碼器選擇「延遲判斷」：不在 import 當下決定，而是等第一次真正要建編碼器時才判斷並快取。
# 原因：main.py 是先 import 本模組、之後才在 main() 裡呼叫 Gst.init(None)。若在 import 當下就判斷，
# 會早於 Gst.init()，此時 GStreamer 尚未初始化、抓不到 NVENC，導致「不設環境變數就誤退 CPU」。
_USE_CPU_ENCODER = None   # None=尚未判斷；True/False=已快取


def use_cpu_encoder():
    """回傳是否使用 CPU 編碼；第一次呼叫時才判斷並快取（此時 Gst.init() 已完成，能正確抓到 NVENC）。"""
    global _USE_CPU_ENCODER
    if _USE_CPU_ENCODER is None:
        _USE_CPU_ENCODER = _detect_cpu_encoder()
    return _USE_CPU_ENCODER


def resolve_tracker_lib():
    """
    自動解析 nvtracker 的 ll-lib-file 路徑（跨平台）。

    順序：
      1. 環境變數 DS_TRACKER_LIB（若指定且存在）
      2. 依序搜尋常見安裝路徑，回傳第一個存在的
      3. 都找不到 → 回傳標準 NVIDIA 路徑（以 nvidia 為主，讓 DS 自行報錯提示）
    """
    env = os.environ.get("DS_TRACKER_LIB", "").strip()
    if env and os.path.exists(env):
        return env
    candidates = [
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
        "/opt/nvidia/deepstream/deepstream-7.0/lib/libnvds_nvmultiobjecttracker.so",
    ]
    # 再補上 glob 掃描（版本號未知時）
    import glob as _glob
    candidates += sorted(_glob.glob(
        "/opt/nvidia/deepstream/deepstream*/lib/libnvds_nvmultiobjecttracker.so"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


def cb_decodebin_child_added(child_proxy, obj, name, user_data):
    """
    nvurisrcbin 內部子元件建立時的回呼（取代 uridecodebin 的 source-setup 訊號，
    nvurisrcbin 沒有 source-setup，須用 child-added 遞迴往內抓）。
      - 內層還有 decodebin 時繼續往下掛，才追得到最底層的 rtspsrc。
      - 對 rtspsrc 強制 TCP、設抖動緩衝與連線逾時、超延遲丟幀。
    只在元件確實有該屬性時才設定（_safe_set），檔案來源的內部元件不受影響。
    """
    if name.find("decodebin") != -1:
        obj.connect("child-added", cb_decodebin_child_added, user_data)
    if name.find("source") != -1:
        _safe_set(obj, "protocols", 4)          # 4 = TCP
        _safe_set(obj, "latency", 200)          # 抖動緩衝 200ms
        _safe_set(obj, "timeout", 5000000)      # 連線逾時 5 秒（微秒）
        _safe_set(obj, "drop-on-latency", True)


def make_elm(gst_type, name):
    """建立 GStreamer 元件；失敗直接結束程式並提示型別/名稱。"""
    elm = Gst.ElementFactory.make(gst_type, name)
    if not elm: sys.exit(f"[ERROR] 無法建立 element: {gst_type} ({name})")
    return elm


def _is_jetson():
    """判斷是否為 Jetson（aarch64 或存在 /etc/nv_tegra_release）。"""
    import os, platform
    return (platform.machine() == "aarch64") or os.path.isfile("/etc/nv_tegra_release")


def _safe_set(elm, name, value):
    """只有當元件確實具有該屬性時才設定，避免跨平台屬性差異造成例外。回傳是否有設成功。"""
    if elm.find_property(name) is not None:
        elm.set_property(name, value)
        return True
    return False


def _configure_encoder(encoder, bitrate, iframeinterval):
    """設定 NVENC 硬體編碼器屬性（Jetson 與 dGPU 的屬性名不同，用 _safe_set 兼容）。"""
    _safe_set(encoder, "bitrate", bitrate)
    _safe_set(encoder, "iframeinterval", iframeinterval)
    _safe_set(encoder, "profile", 0)
    is_jetson_enc = False
    # 下列屬性多為 Jetson NVENC 專有；若有設成功代表是 Jetson 編碼器
    is_jetson_enc |= _safe_set(encoder, "preset-level", 1)
    is_jetson_enc |= _safe_set(encoder, "insert-sps-pps", 1)
    is_jetson_enc |= _safe_set(encoder, "maxperf-enable", 1)
    if not is_jetson_enc:
        # dGPU NVENC 的調校屬性
        _safe_set(encoder, "preset-id", 1)
        _safe_set(encoder, "tuning-info-id", 2)


def _enc_caps_string(framerate=None):
    """依編碼器型別回傳編碼前所需的 caps 字串（CPU 走 I420 系統記憶體；NVENC 走 NV12 NVMM）。"""
    base = "video/x-raw, format=I420" if use_cpu_encoder() else "video/x-raw(memory:NVMM), format=NV12"
    if framerate is not None: base += f", framerate={framerate}/1"
    return base


def _make_encoder(name_prefix, i, codec, bitrate_bps, iframeinterval):
    """
    依 USE_CPU_ENCODER 與 codec 建立對應編碼器：
      CPU：x264enc / x265enc（bitrate 單位為 kbps）。
      GPU：nvv4l2h264enc / nvv4l2h265enc。
    """
    is_h265 = (codec == "h265")
    if use_cpu_encoder():
        enc_type = "x265enc" if is_h265 else "x264enc"
        encoder = make_elm(enc_type, f"{name_prefix}-{i}")
        _safe_set(encoder, "bitrate", max(1, int(bitrate_bps / 1000)))  # bps → kbps
        _safe_set(encoder, "speed-preset", 1)   # 1=ultrafast，吞吐優先
        _safe_set(encoder, "tune", 4)           # 4=zerolatency
        _safe_set(encoder, "key-int-max", iframeinterval)
    else:
        enc_type = "nvv4l2h265enc" if is_h265 else "nvv4l2h264enc"
        encoder = make_elm(enc_type, f"{name_prefix}-{i}")
        _configure_encoder(encoder, bitrate=bitrate_bps, iframeinterval=iframeinterval)
    return encoder


def _get_tile_layout(num_sources):
    """依來源數決定顯示拼接的列/欄與整體寬高（cell 維持 16:9）。"""
    if num_sources == 1: rows, cols = 1, 1
    elif num_sources == 2: rows, cols = 1, 2
    elif num_sources <= 4: rows, cols = 2, 2
    elif num_sources <= 6: rows, cols = 2, 3
    elif num_sources <= 9: rows, cols = 3, 3
    else: rows, cols = 4, 4
    total_width = 1920
    cell_w = total_width // cols
    cell_h = int(cell_w * 9 / 16)
    total_height = cell_h * rows
    return rows, cols, total_width, total_height


def _build_save_branch_for_file(pipeline, pad_index, video_path, source_fps):
    """
    建立「檔案來源」的存檔分支：nvvideoconvert → videorate → caps → 編碼器 → parse → qtmux → filesink。
    回傳此分支的入口元件（nvvideoconvert），供上游 link。
    """
    i = pad_index
    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)
    videorate = make_elm("videorate", f"videorate-save-{i}")
    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps)) if int(round(source_fps)) > 0 else 30
    cap_filter.set_property("caps", Gst.Caps.from_string(_enc_caps_string(framerate=fps_int)))
    encoder = _make_encoder("encoder", i, codec="h264", bitrate_bps=4000000, iframeinterval=fps_int)
    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)
    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]: pipeline.add(elm)
    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)
    return nvvidconv_s


def _build_save_branch_for_rtsp(pipeline, pad_index, video_path, source_fps):
    """
    建立「RTSP/即時來源」的存檔分支（結構同檔案版，videorate 命名略異）。
    回傳入口元件（nvvideoconvert）。
    """
    i = pad_index
    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)
    videorate = make_elm("videorate", f"videorate-{i}")
    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps)) if int(round(source_fps)) > 0 else 30
    cap_filter.set_property("caps", Gst.Caps.from_string(_enc_caps_string(framerate=fps_int)))
    encoder = _make_encoder("encoder", i, codec="h264", bitrate_bps=4000000, iframeinterval=fps_int)
    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)
    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]: pipeline.add(elm)
    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)
    return nvvidconv_s


def _build_display_sink(pipeline, num_sources, has_live_source=False):
    """
    建立顯示分支：第二顆 nvstreammux（重新批次）→ nvmultistreamtiler（拼接）
    → nvvideoconvert →（Jetson 視情況加 nvegltransform）→ 顯示 sink。
    回傳此顯示用 streammux，供每路的 show 分支 link 各自的 sink_i。
    """
    rows, cols, total_w, total_h = _get_tile_layout(num_sources)
    is_jetson = _is_jetson()
    # 只有 Jetson 且有 nvegltransform 才使用（dGPU/WSL 不需要）
    use_egltransform = is_jetson and (Gst.ElementFactory.find("nvegltransform") is not None)
    streammux2 = make_elm("nvstreammux", "Stream-muxer-display")
    streammux2.set_property("batch-size", num_sources)  # 新舊版 mux 皆支援

    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改用 config_mux.txt。
        # 來源皆 1080p、真正拼接由 tiler 完成，故新版 mux 不縮放也不影響顯示結果。
        _mux_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_mux.txt"
        )
        if os.path.exists(_mux_cfg):
            streammux2.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，顯示用 mux 將用內建預設值")
    else:
        # 舊版 mux：維持原本設定
        streammux2.set_property("width", 1920)
        streammux2.set_property("height", 1080)
        streammux2.set_property("batched-push-timeout", 10000)
        # ⭐ live-source 依主線是否有 live 來源決定，不可寫死 1
        streammux2.set_property("live-source", 1 if has_live_source else 0)
        streammux2.set_property("nvbuf-memory-type", 0)
    tiler = make_elm("nvmultistreamtiler", "nvtiler-display")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", total_w)
    tiler.set_property("height", total_h)
    q_d1 = make_elm("queue", "q-display-1")
    nvvidconv = make_elm("nvvideoconvert", "convertor-display")
    nvvidconv.set_property("nvbuf-memory-type", 0)
    q_d2 = make_elm("queue", "q-display-2")
    q_d3 = make_elm("queue", "q-display-3")

    # 先找 NVIDIA 專用 sink（吃 NVMM，可直送）；dGPU/純 WSLg 常沒有，回傳 None 走路徑 B
    nv_sink = None
    if Gst.ElementFactory.find("nveglglessink") is not None:
        nv_sink = make_elm("nveglglessink", "nvvideo-renderer-display")
    elif Gst.ElementFactory.find("nv3dsink") is not None:
        nv_sink = make_elm("nv3dsink", "nvvideo-renderer-display")
        use_egltransform = False

    if nv_sink is not None:
        # ---- 路徑 A：NVIDIA 專用 sink（走 NVMM，Jetson 視情況加 nvegltransform）----
        sink = nv_sink
        sink.set_property("sync", False)
        _safe_set(sink, "qos", False)
        if use_egltransform:
            transform = make_elm("nvegltransform", "nvegl-transform-display")
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, transform, q_d3, sink]
        else:
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, q_d3, sink]
        for elm in elements: pipeline.add(elm)
        streammux2.link(tiler)
        tiler.link(q_d1)
        q_d1.link(nvvidconv)
        nvvidconv.link(q_d2)
        if use_egltransform:
            q_d2.link(transform)
            transform.link(q_d3)
        else:
            q_d2.link(q_d3)
        q_d3.link(sink)
    else:
        # ---- 路徑 B：標準 GStreamer sink（dGPU / 純 WSLg / 無 NVIDIA sink）----
        # nvvideoconvert 先把畫面從 NVMM 轉到系統記憶體，再交給標準 sink 顯示。
        # 可用環境變數 DS_DISPLAY_SINK 指定要用哪個標準 sink（例如 ximagesink）。
        caps_sys = make_elm("capsfilter", "caps-display-sys")
        caps_sys.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        videoconv = make_elm("videoconvert", "videoconvert-display")

        forced = os.environ.get("DS_DISPLAY_SINK", "").strip()
        candidates = [forced] if forced else ["ximagesink", "glimagesink", "autovideosink"]
        std_sink = None
        for cand in candidates:
            if cand and Gst.ElementFactory.find(cand) is not None:
                std_sink = make_elm(cand, "nvvideo-renderer-display")
                print(f"[INFO] 使用標準顯示 sink：{cand}")
                break
        if std_sink is None:
            sys.exit(f"[ERROR] 找不到可用的顯示 sink（嘗試清單：{candidates}）")
        _safe_set(std_sink, "sync", False)
        sink = std_sink

        elements = [streammux2, tiler, q_d1, nvvidconv, caps_sys, videoconv, q_d2, sink]
        for elm in elements: pipeline.add(elm)
        streammux2.link(tiler)
        tiler.link(q_d1)
        q_d1.link(nvvidconv)
        nvvidconv.link(caps_sys)
        caps_sys.link(videoconv)
        videoconv.link(q_d2)
        q_d2.link(sink)
    return streammux2


def setup_cam_branch(pipeline, pad_index, cfg, demux, display_streammux, osd_probe_callback):
    """
    demux 之後、為「單一路」組裝下游：
        demux.src_i → queue → nvvideoconvert(RGBA NVMM) → nvdsosd
                    → 依設定接 顯示 / 存檔 / RTSP（可同時多個，用 tee 分流）

    並在 nvosd.sink 掛上 per_cam_osd_probe（疊 FPS 文字）。
    截圖「不在這裡做」——影像已在 streammux 前由 appsink 取得（見 main.cb_newpad）。

    參數：
        pipeline:              GstPipeline
        pad_index (int):       哪一路
        cfg (dict):            該路設定
        demux:                 nvstreamdemux
        display_streammux:     顯示用 streammux（無顯示時為 None）
        osd_probe_callback:    per_cam_osd_probe
    """
    i = pad_index
    src_pad = demux.get_request_pad(f"src_{i}")

    # ⭐ 最乾淨的 Pipeline：沒有任何截圖分流（截圖已在 streammux 前處理）
    q_cam = make_elm("queue", f"q-cam-{i}")
    nvvidconv_osd = make_elm("nvvideoconvert", f"conv_osd_{i}")
    nvvidconv_osd.set_property("nvbuf-memory-type", 0)
    caps_osd = make_elm("capsfilter", f"caps_osd_{i}")
    caps_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd_i = make_elm("nvdsosd", f"nvosd-{i}")
    # process-mode：Jetson 用 2（VIC/HW），dGPU/WSL 用 1（GPU）
    nvosd_i.set_property("process-mode", 2 if _is_jetson() else 1)

    for elm in [q_cam, nvvidconv_osd, caps_osd, nvosd_i]: pipeline.add(elm)
    src_pad.link(q_cam.get_static_pad("sink"))
    q_cam.link(nvvidconv_osd)
    nvvidconv_osd.link(caps_osd)
    caps_osd.link(nvosd_i)

    # 在 nvosd.sink 掛 FPS overlay 探針（帶入本路 index）
    nvosd_i.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info, idx=i: osd_probe_callback(pad, info, idx),
        0
    )

    # ---- 依設定決定要接哪些輸出分支 ----
    cam_save = cfg.get("output", {}).get("save_output_video", False)
    cam_show = cfg.get("display", {}).get("show_window", True)
    is_file = cfg.get("is_file_source", False)
    enabled_branches = sum([cam_save, cam_show])

    # 情況 0：完全沒有輸出 → 接 fakesink 收尾（仍需消化 buffer）
    if enabled_branches == 0:
        fake = make_elm("fakesink", f"fake-{i}")
        fake.set_property("sync", False)
        fake.set_property("async", False)
        pipeline.add(fake)
        nvosd_i.link(fake)
        return

    # 情況 1：只有一種輸出 → nvosd 直接接該分支（不需 tee）
    if enabled_branches == 1:
        if cam_save:
            if is_file: nvosd_i.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
            else: nvosd_i.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        elif cam_show: _link_show_branch(pipeline, i, nvosd_i, display_streammux)
        return

    # 情況 2：存檔 + 顯示同時 → nvosd 接 tee，再分給各分支
    tee = make_elm("tee", f"tee-{i}")
    pipeline.add(tee)
    nvosd_i.link(tee)

    if cam_save:
        q_s = make_elm("queue", f"q-s-{i}")
        pipeline.add(q_s)
        tee.link(q_s)
        if is_file: q_s.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        else: q_s.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))

    if cam_show: _link_show_branch(pipeline, i, tee, display_streammux)


def _link_show_branch(pipeline, i, upstream, display_streammux):
    """把某一路接到顯示用 streammux 的 sink_i：upstream → queue → nvvideoconvert → display_streammux.sink_i。"""
    q_d = make_elm("queue", f"q-d-{i}")
    nv_d = make_elm("nvvideoconvert", f"nv-d-{i}")
    nv_d.set_property("nvbuf-memory-type", 0)
    pipeline.add(q_d)
    pipeline.add(nv_d)
    upstream.link(q_d)
    q_d.link(nv_d)
    nv_d.get_static_pad("src").link(display_streammux.get_request_pad(f"sink_{i}"))
