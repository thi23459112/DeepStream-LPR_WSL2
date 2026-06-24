#!/usr/bin/env python3
"""
================================================================================
 main.py — DeepStream 7.1 多路車牌辨識 (LPR) 主程式
================================================================================
整體 pipeline：

  [每路 source] uridecodebin ─ tee ┬─► (主推論) nvstreammux
                                    └─► (截圖)   queue→nvvideoconvert→appsink

  nvstreammux → q1 → nvdspreprocess → q2 → pgie(車輛) → q3
              → q_sgie_plate → sgie_plate(車牌) → q_sgie_num → sgie_num(字元)
              → q_analytics → nvdsanalytics → q4 → nvstreamdemux
              → (每路) queue → nvvideoconvert → nvdsosd → 顯示/存檔/RTSP

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
import select
import termios
import tty
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer

import pyds

from logic.color import load_labels, CLASS_MAP
from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
    INFER_SEC_PLATE_CONFIG, INFER_SEC_NUM_CONFIG,
    TRACKER_MODE, BOXMOT_TRACKER_CONFIG,
)
from logic.state_db import initialize_state_managers, force_finalize_all
from logic.pipeline import (
    cb_source_setup, make_elm,
    _build_display_sink, setup_cam_branch,
)
from logic.probes import (
    tracker_src_pad_buffer_probe,
    boxmot_pgie_src_probe,
    expand_plate_probe,
    assemble_plate_probe,
    per_cam_osd_probe,
    set_obj_enc_context,
    g_frame_buffer,         # ⭐ 全域幀緩存（依 PTS 對齊）
    store_frame_for_pad,    # ⭐ appsink 用：把幀連同 PTS 存入緩存
)

# ---- 全域物件（供 keyboard / bus / 退出流程共用）----
g_loop          = None
g_pipeline      = None
g_eos_triggered = False
g_rtsp_server   = None
g_obj_enc_ctx   = None


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
def cb_newpad(decodebin, decoder_src_pad, data):
    """
    uridecodebin 解碼出視訊 pad 時觸發。用 tee 把解碼後畫面分成兩路：
      分支 A：接到 nvstreammux（主推論線）。
      分支 B：（若該路啟用截圖）queue→nvvideoconvert(RGBA,系統記憶體)→appsink。

    為何在 streammux 之前分流：這裡拿到的是「乾淨、未疊 OSD、且每路獨立」的畫面，
    且不受 demux 後 per-source 取像限制；appsink 走 leaky/drop，不會反壓主推論。
    """
    caps = decoder_src_pad.get_current_caps()
    if not caps: caps = decoder_src_pad.query_caps()
    if caps.get_structure(0).get_name().find("video") != -1:
        pad_index = data["pad_index"]
        streammux = data["streammux"]
        pipeline = data["pipeline"]
        cfg = data["cfg"]

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


def _start_rtsp_server(rtsp_routes):
    """
    依各路的 RTSP 推流設定，建立 GstRtspServer 並掛載各 mount point。
    同一個 port 下可掛多條路徑（依 udp_port 區分來源）。
    """
    if not rtsp_routes: return None
    routes_by_port = {}
    for r in rtsp_routes: routes_by_port.setdefault(r["port"], []).append(r)
    servers = []
    for port, routes in routes_by_port.items():
        server = GstRtspServer.RTSPServer()
        server.set_service(str(port))
        mounts = server.get_mount_points()
        for r in routes:
            udp_port, encoder = r["udp_port"], r["encoder"]
            mount_path = "/" + r["mount_path"].lstrip("/")
            enc_name = "H265" if encoder == "h265" else "H264"
            # 從 udpsink 推出的 RTP 由 udpsrc 接回，再 depay/pay 給 RTSP client
            launch_str = (f"( udpsrc port={udp_port} caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name={enc_name}, payload=96\" "
                          f"! rtp{encoder}depay ! rtp{encoder}pay name=pay0 pt=96 )")
            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(launch_str)
            factory.set_shared(True)
            mounts.add_factory(mount_path, factory)
            print(f"[INFO] RTSP 推流註冊: rtsp://<本機IP>:{port}{mount_path}")
        server.attach(None)
        servers.append(server)
    return servers


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
    global g_loop, g_pipeline, g_eos_triggered, g_rtsp_server, g_obj_enc_ctx

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
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 70000)
    streammux.set_property("live-source", 1)
    streammux.set_property("nvbuf-memory-type", 0)
    g_pipeline.add(streammux)

    # ---- 各路 source：uridecodebin，pad-added 時在 cb_newpad 內分流 ----
    for pad_index, cfg in SOURCE_CONFIGS.items():
        source = make_elm("uridecodebin", f"uri-decode-bin-{pad_index}")
        source.set_property("uri", cfg["source"])
        # ⭐ 傳入更多參數給 cb_newpad（streammux/pad_index/pipeline/cfg）
        source.connect("pad-added", cb_newpad, {
            "streammux": streammux,
            "pad_index": pad_index,
            "pipeline": g_pipeline,
            "cfg": cfg
        })
        source.connect("source-setup", cb_source_setup, None)
        g_pipeline.add(source)

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
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
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
    rtsp_routes = []
    for pad_index, cfg in SOURCE_CONFIGS.items():
        udp_port = setup_cam_branch(g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe)
        if udp_port is not None:
            rtsp_routes.append({"pad_index": pad_index, "udp_port": udp_port, "port": cfg["rtsp_push"]["port"], "mount_path": cfg["rtsp_push"]["mount_path"], "encoder": cfg["rtsp_push"]["encoder"]})

    if rtsp_routes:
        g_rtsp_server = _start_rtsp_server(rtsp_routes)
        print(f"[INFO] 共 {len(rtsp_routes)} 條 RTSP 推流就緒")

    # ---- 啟動：鍵盤監看 + Bus 監看 + MainLoop ----
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)   # 不需 Enter 即可讀到單鍵
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按下 'q' 鍵即可優雅退出並存檔...\n")
        g_loop = GLib.MainLoop()
        bus = g_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, g_loop)
        g_pipeline.set_state(Gst.State.PLAYING)
        g_loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，準備發送 EOS...")
        if not g_eos_triggered:
            g_eos_triggered = True
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(8, force_quit_loop)
        try: g_loop.run()
        except KeyboardInterrupt: pass
    finally:
        # ---- 收尾：還原終端機設定、強制結算、釋放資源 ----
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        force_finalize_all()
        g_pipeline.set_state(Gst.State.NULL)
        if g_obj_enc_ctx is not None:
            try: pyds.nvds_obj_enc_destroy_context(g_obj_enc_ctx)
            except Exception as e: print(f"[WARNING] 銷毀 Object Encoder context 失敗：{e}")


if __name__ == '__main__':
    initialize_state_managers()
    main()