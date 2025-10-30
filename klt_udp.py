import os
os.environ['GST_PLUGIN_PATH'] = os.environ.get('GST_PLUGIN_PATH', '/usr/lib/aarch64-linux-gnu/gstreamer-1.0')

gi_typelib_paths = [
    '/usr/lib/aarch64-linux-gnu/girepository-1.0',
    '/usr/lib/girepository-1.0',
]
existing_path = os.environ.get('GI_TYPELIB_PATH', '')
os.environ['GI_TYPELIB_PATH'] = ':'.join(gi_typelib_paths + [existing_path] if existing_path else gi_typelib_paths)

ld_library_paths = [
    '/usr/lib/aarch64-linux-gnu',
    '/usr/lib',
]
existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_library_paths + [existing_ld_path] if existing_ld_path else ld_library_paths)

import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unknown"

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30
DEBUG_DRAW   = False

ROI_SIZE = 120
MIN_POINTS = 2
MAX_CORNERS = 10
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
GFTT_BLOCKSIZE = 7

LK_FRAME_SCALE = 0.3

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

ROI_MOVE_ALPHA = 1
MIN_POINTS_FOR_MOVE = 0
LOCK_INITIAL_FEATURES = True

UDP_HOST = '192.168.144.61'
UDP_PORT = 5001

RTSP_PORT = 8554
RTSP_MOUNT_POINT = '/video0'
BITRATE_KBPS = 4500

SERIAL_CANDIDATES = [
    '/dev/serial0',
    '/dev/ttyAMA10'
]
SERIAL_BAUD = 115200

latest_point = None
new_point_received = False
target_selected = False
zoom_level = 1.0
zoom_command = None
zoom_center = None
tracking = False
prev_frame = None
tracking_points = None
roi_rect = None
roi_center = None
adaptive_mode = True
feature_lock_active = False

serial_port = None

def display_to_original_coord(x, y):
    return int(x), int(y)

def setup_serial():
    for dev in SERIAL_CANDIDATES:
        try:
            ser = serial.Serial(
                port=dev,
                baudrate=SERIAL_BAUD,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            return ser
        except Exception:
            continue
    return None

def normalize_for_serial(px, py, width, height):
    xn = int((-1.0 + (px * (2.0 / max(1, width)))) * 1000.0)
    yn = int(( 1.0 - (py * (2.0 / max(1, height)))) * 1000.0)
    return xn, yn

def send_data_to_serial(px, py, is_tracking, frame_w, frame_h):
    global serial_port
    if serial_port is None or not serial_port.is_open:
        return

    x_norm, y_norm = normalize_for_serial(px, py, frame_w, frame_h)

    try:
        data = struct.pack('<BBhhB', 0xBB, 0x88, x_norm, y_norm, 0xFF if is_tracking else 0x00)
        checksum = 0
        for b in data:
            checksum ^= b
        data += bytes([checksum])
        serial_port.write(data)
    except Exception:
        try:
            if serial_port and serial_port.is_open:
                serial_port.close()
            time.sleep(0.25)
            serial_port = setup_serial()
        except Exception:
            pass

def gstreamer_camera_pipeline(width, height, fps):
    pipelines = [
        f"v4l2src device=/dev/video0 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
        f"v4l2src device=/dev/video1 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
        f"libcamerasrc ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    ]
    return pipelines

class RTSPServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps, bitrate_kbps):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.appsrc_list = []
        self.client_connected = False
        self.frame_count = 0
        self.start_time = None
        self.frame_duration_ns = int(1_000_000_000 / fps)

        pipeline = (
            f"appsrc name=source is-live=true format=3 do-timestamp=false "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={bitrate_kbps} tune=zerolatency speed-preset=ultrafast key-int-max=24 bframes=0 "
            f"rc-lookahead=0 sync-lookahead=0 sliced-threads=true ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"rtph264pay name=pay0 pt=96 config-interval=1 aggregate-mode=zero-latency"
        )
        self.set_launch(pipeline)
        self.set_shared(True)

        self.connect("media-configure", self.on_media_configure)

    def on_media_configure(self, factory, media):
        element = media.get_element()
        if element:
            appsrc = element.get_by_name("source")
            if appsrc:
                self.appsrc_list.append(appsrc)
                self.client_connected = True

                appsrc.set_property('format', Gst.Format.TIME)
                appsrc.set_property('block', False)
                appsrc.set_property('max-bytes', 0)
                appsrc.set_property('emit-signals', False)

    def push_frame(self, frame):
        if not self.appsrc_list:
            return
        
        # PTS 계산
        if self.start_time is None:
            self.start_time = time.time()
        
        pts = self.frame_count * self.frame_duration_ns
        
        # Dead appsrc 제거하면서 프레임 푸시
        active_appsrc = []
        for appsrc in self.appsrc_list:
            try:
                # 버퍼 생성
                buf = Gst.Buffer.new_wrapped(frame.tobytes())
                buf.pts = pts
                buf.duration = self.frame_duration_ns
                
                # 프레임 푸시
                ret = appsrc.emit('push-buffer', buf)
                
                if ret == Gst.FlowReturn.OK:
                    active_appsrc.append(appsrc)
                # ret가 FLUSHING이나 EOS면 연결 끊김
            except Exception as e:
                print(f"Failed to push frame to appsrc: {e}")
        
        # 리스트 업데이트
        self.appsrc_list = active_appsrc
        self.client_connected = len(self.appsrc_list) > 0
        
        self.frame_count += 1

    @property
    def appsrc(self):
        return self.appsrc_list[0] if self.appsrc_list else None

def on_client_connected(server, client):
    pass

def setup_rtsp_server(port, mount_point, width, height, fps, bitrate_kbps):
    server = GstRtspServer.RTSPServer()

    server.set_address("0.0.0.0")

    server.set_service(str(port))

    factory = RTSPServerFactory(width, height, fps, bitrate_kbps)

    mounts = server.get_mount_points()
    mounts.add_factory(mount_point, factory)

    server.connect("client-connected", on_client_connected)

    server.attach(None)

    local_ip = get_local_ip()

    return server, factory

def clamp_roi(cx, cy, w, h, W, H):
    x = int(cx - w // 2)
    y = int(cy - h // 2)
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return x, y, w, h

def detect_points_in_roi(gray, rect):
    x, y, w, h = rect
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    pts = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        mask=None,
        blockSize=GFTT_BLOCKSIZE,
        useHarrisDetector=False
    )

    if pts is None:
        return None

    offset = np.array([[[x, y]]], dtype=np.float32)
    return pts + offset

def in_roi(pts, rect):
    if pts is None or len(pts) == 0:
        return np.array([], dtype=bool)
    x, y, w, h = rect
    xs = pts[:, 0]
    ys = pts[:, 1]
    return (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)

def calculate_centroid(pts):
    if pts is None or len(pts) == 0:
        return None
    return np.mean(pts.reshape(-1, 2), axis=0)

def update_roi_position(roi_rect, target_center, W, H, alpha=ROI_MOVE_ALPHA):
    x, y, w, h = roi_rect
    current_cx = x + w // 2
    current_cy = y + h // 2
    
    new_cx = current_cx + alpha * (target_center[0] - current_cx)
    new_cy = current_cy + alpha * (target_center[1] - current_cy)
    
    return clamp_roi(new_cx, new_cy, w, h, W, H)

def udp_receiver():
    global latest_point, new_point_received, target_selected, zoom_command, tracking, zoom_center, feature_lock_active

    prev_x = prev_y = None
    prev_target_selected = None
    prev_zoom_cmd = None

    max_retries, retry_delay = 30, 1.0
    for attempt in range(max_retries):
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.settimeout(0.5)
            udp_socket.bind((UDP_HOST, UDP_PORT))
            break
        except Exception:
            time.sleep(retry_delay)
    else:
        return

    try:
        while True:
            try:
                data, _ = udp_socket.recvfrom(8)
            except socket.timeout:
                continue
            except Exception:
                break
            if len(data) < 8 or data[0] != 0xAA or data[1] != 0x77:
                continue
            x = struct.unpack('<H', data[2:4])[0]
            y = struct.unpack('<H', data[4:6])[0]
            is_target = (data[6] == 0xFF)
            zoom_cmd = data[7]
            if (x == prev_x and y == prev_y and is_target == prev_target_selected and zoom_cmd == prev_zoom_cmd):
                continue
            prev_x, prev_y = x, y
            prev_target_selected = is_target
            prev_zoom_cmd = zoom_cmd
            orig_x, orig_y = x, y
            try:
                orig_x, orig_y = display_to_original_coord(x, y)
            except Exception:
                pass
            if is_target:
                latest_point = (orig_x, orig_y)
                new_point_received = True
                target_selected = True
                if zoom_center is None:
                    zoom_center = (orig_x, orig_y)
            elif data[6] == 0x00 and target_selected:
                target_selected = False
                tracking = False
                feature_lock_active = False
            if zoom_cmd == 0x02 and zoom_command != 'zoom_in':
                zoom_command = 'zoom_in'
                if zoom_center is None:
                    zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x01 and zoom_command != 'zoom_out':
                zoom_command = 'zoom_out'
                if zoom_center is None:
                    zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x00:
                zoom_command = None

    finally:
        try:
            udp_socket.close()
        except Exception:
            pass

def process_new_coordinate(gray_frame):
    global latest_point, new_point_received, tracking, prev_frame, tracking_points, roi_rect, roi_center
    global feature_lock_active

    if not new_point_received:
        return

    new_point_received = False
    x, y = latest_point

    if not (0 <= x < gray_frame.shape[1] and 0 <= y < gray_frame.shape[0]):
        return

    H, W = gray_frame.shape[:2]
    roi_rect = clamp_roi(x, y, ROI_SIZE, ROI_SIZE, W, H)
    roi_center = (x, y)
    
    tracking_points = detect_points_in_roi(gray_frame, roi_rect)
    
    if tracking_points is not None and len(tracking_points) > 0:
        prev_frame = gray_frame
        tracking = True
        feature_lock_active = LOCK_INITIAL_FEATURES
    else:
        feature_lock_active = False

def main():
    global serial_port, tracking, prev_frame, tracking_points, roi_rect, roi_center
    global zoom_level, zoom_command, zoom_center, target_selected, display_to_original_coord, feature_lock_active

    try:
        Gst.init(None)
    except Exception as e:
        return

    serial_port = setup_serial()

    mainloop = GLib.MainLoop()
    mainloop_thread = threading.Thread(target=mainloop.run, daemon=True)
    mainloop_thread.start()

    rtsp_server, rtsp_factory = setup_rtsp_server(
        RTSP_PORT, RTSP_MOUNT_POINT,
        FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, BITRATE_KBPS
    )

    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()

    cap_pipelines = gstreamer_camera_pipeline(FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS)
    cap = None

    for i, pipeline in enumerate(cap_pipelines):
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                break
            else:
                cap.release()
                cap = None
        except Exception as e:
            if cap:
                cap.release()
                cap = None

    if cap is None:
        for device_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 24)

                    ret, test_frame = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            except Exception as e:
                if cap:
                    cap.release()
                    cap = None

    if cap is None:
        raise RuntimeError("Failed to open camera with any method")

    appsrc = None

    last_serial_time = 0.0
    serial_interval = 1.0 / float(TARGET_FPS)
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame_counter += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level += 0.5
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level -= 0.5
                zoom_command = None

            process_new_coordinate(gray)

            center_x = center_y = 0
            if not target_selected:
                tracking = False
                feature_lock_active = False

            if tracking and tracking_points is not None and len(tracking_points) > 0 and prev_frame is not None:
                H, W = gray.shape[:2]
                if roi_rect is not None:
                    rx, ry, rw, rh = roi_rect
                else:
                    rx, ry, rw, rh = 0, 0, W, H

                rx = max(0, min(rx, W - 1))
                ry = max(0, min(ry, H - 1))
                rx2 = min(rx + rw, W)
                ry2 = min(ry + rh, H)
                roi_width = max(0, rx2 - rx)
                roi_height = max(0, ry2 - ry)

                p1 = None
                st = err = None

                if roi_width > 0 and roi_height > 0:
                    prev_roi = prev_frame[ry:ry2, rx:rx2]
                    gray_roi = gray[ry:ry2, rx:rx2]

                    if prev_roi.size > 0 and gray_roi.size > 0:
                        offset = np.array([[[rx, ry]]], dtype=np.float32)
                        local_points = tracking_points - offset

                        p1_local = None

                        if LK_FRAME_SCALE < 1.0 and roi_width > 1 and roi_height > 1:
                            scaled_w = max(1, int(roi_width * LK_FRAME_SCALE))
                            scaled_h = max(1, int(roi_height * LK_FRAME_SCALE))
                            prev_scaled = cv2.resize(prev_roi, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                            gray_scaled = cv2.resize(gray_roi, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                            scaled_points = local_points * LK_FRAME_SCALE
                            p1_local, st, err = cv2.calcOpticalFlowPyrLK(
                                prev_scaled, gray_scaled, scaled_points, None, **LK_PARAMS
                            )
                            if p1_local is not None:
                                p1_local = p1_local / LK_FRAME_SCALE
                        else:
                            p1_local, st, err = cv2.calcOpticalFlowPyrLK(
                                prev_roi, gray_roi, local_points, None, **LK_PARAMS
                            )

                        if p1_local is not None:
                            p1 = p1_local + offset
                
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = tracking_points[st == 1]

                    if adaptive_mode and len(good_new) >= MIN_POINTS_FOR_MOVE:
                        centroid = calculate_centroid(good_new)
                        if centroid is not None:
                            roi_rect = update_roi_position(roi_rect, centroid, W, H, ROI_MOVE_ALPHA)
                            
                            mask_in = in_roi(good_new, roi_rect)
                            good_new = good_new[mask_in]
                            good_old = good_old[mask_in]
                    else:
                        if roi_rect is not None:
                            mask_in = in_roi(good_new, roi_rect)
                            good_new = good_new[mask_in]
                            good_old = good_old[mask_in]

                    if len(good_new) > 0:
                        tracking_points = good_new.reshape(-1, 1, 2).astype(np.float32)
                        centroid = calculate_centroid(good_new)
                        if centroid is not None:
                            center_x, center_y = centroid.astype(int)
                    else:
                        tracking_points = None
                        tracking = False
                        feature_lock_active = False

                    if (tracking_points is None or len(tracking_points) < MIN_POINTS) and roi_rect is not None:
                        if not (LOCK_INITIAL_FEATURES and feature_lock_active):
                            tracking_points = detect_points_in_roi(gray, roi_rect)
                            if tracking_points is not None and len(tracking_points) > 0:
                                feature_lock_active = LOCK_INITIAL_FEATURES

                prev_frame = gray

            display_frame = frame
            zoom_x1 = zoom_y1 = 0
            zoom_applied = False

            if zoom_level > 1.0 and zoom_center is not None:
                h, w = display_frame.shape[:2]
                cx, cy = zoom_center
                zw = int(w / zoom_level)
                zh = int(h / zoom_level)

                zoom_x1 = max(0, min(w - zw, int(cx - zw // 2)))
                zoom_y1 = max(0, min(h - zh, int(cy - zh // 2)))

                roi = display_frame[zoom_y1:zoom_y1 + zh, zoom_x1:zoom_x1 + zw]
                display_frame = cv2.resize(roi, (w, h))
                zoom_applied = True

            def original_to_display_coord(ox, oy):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(ox), int(oy)
                rel_x = ox - zoom_x1
                rel_y = oy - zoom_y1
                return int(rel_x * zoom_level), int(rel_y * zoom_level)

            def local_display_to_original_coord(dx, dy):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(dx), int(dy)
                rel_x = dx / zoom_level
                rel_y = dy / zoom_level
                ox = int(rel_x + zoom_x1)
                oy = int(rel_y + zoom_y1)
                h, w = frame.shape[:2]
                return max(0, min(ox, w - 1)), max(0, min(oy, h - 1))

            display_to_original_coord = lambda dx, dy: local_display_to_original_coord(dx, dy)

            if tracking and tracking_points is not None:
                for pt in tracking_points:
                    arr = pt.ravel().astype(int)
                    x, y = int(arr[0]), int(arr[1])
                    dx, dy = original_to_display_coord(x, y)
                    cv2.circle(display_frame, (dx, dy), 3, (0, 255, 0), -1)

                cx_d, cy_d = original_to_display_coord(center_x, center_y)
                cv2.circle(display_frame, (cx_d, cy_d), 5, (0, 0, 255), -1)

                now = time.time()
                if now - last_serial_time >= serial_interval:
                    fh, fw = frame.shape[:2]
                    send_data_to_serial(center_x, center_y, tracking, fw, fh)
                    last_serial_time = now

            if roi_rect is not None:
                x, y, w, h = roi_rect
                dx1, dy1 = original_to_display_coord(x, y)
                dx2, dy2 = original_to_display_coord(x+w, y+h)
                color = (0, 255, 255) if adaptive_mode else (255, 128, 0)
                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                
                cx, cy = x + w//2, y + h//2
                dcx, dcy = original_to_display_coord(cx, cy)
                cv2.drawMarker(display_frame, (dcx, dcy), color, cv2.MARKER_CROSS, 10, 1)

            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if tracking else (0, 0, 255), 2)
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level:.1f}x", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if LK_FRAME_SCALE < 1.0:
                cv2.putText(display_frame, f"LK Scale: {LK_FRAME_SCALE:.1f}", (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if rtsp_factory.client_connected:
                cv2.putText(display_frame, "RTSP: STREAMING", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "RTSP: WAITING", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            rtsp_factory.push_frame(display_frame)

            if os.path.exists("stop.signal"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            if serial_port and serial_port.is_open:
                serial_port.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()