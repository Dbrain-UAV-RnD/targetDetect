import os
os.environ['GST_PLUGIN_PATH'] = os.environ.get('GST_PLUGIN_PATH', '/usr/lib/aarch64-linux-gnu/gstreamer-1.0')

# conda 환경에서 GObject Introspection 라이브러리 찾기 위한 설정
gi_typelib_paths = [
    '/usr/lib/aarch64-linux-gnu/girepository-1.0',
    '/usr/lib/girepository-1.0',
]
existing_path = os.environ.get('GI_TYPELIB_PATH', '')
os.environ['GI_TYPELIB_PATH'] = ':'.join(gi_typelib_paths + [existing_path] if existing_path else gi_typelib_paths)

# 라이브러리 경로 설정
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

UDP_HOST = '192.168.10.219'
UDP_PORT = 5001

RTSP_PORT = 544
RTSP_MOUNT_POINT = '/video0'
BITRATE_KBPS = 6000

SERIAL_CANDIDATES = [
    '/dev/serial0',
    '/dev/ttyAMA10'
]
SERIAL_BAUD = 57600

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

crc16_table = [0x0000, 0xc0c1, 0xc181, 0x0140, 0xc301, 0x03c0, 0x0280, 0xc241, 
    0xc601, 0x06c0, 0x0780, 0xc741, 0x0500, 0xc5c1, 0xc481, 0x0440, 
    0xcc01, 0x0cc0, 0x0d80, 0xcd41, 0x0f00, 0xcfc1, 0xce81, 0x0e40, 
    0x0a00, 0xcac1, 0xcb81, 0x0b40, 0xc901, 0x09c0, 0x0880, 0xc841, 
    0xd801, 0x18c0, 0x1980, 0xd941, 0x1b00, 0xdbc1, 0xda81, 0x1a40, 
    0x1e00, 0xdec1, 0xdf81, 0x1f40, 0xdd01, 0x1dc0, 0x1c80, 0xdc41, 
    0x1400, 0xd4c1, 0xd581, 0x1540, 0xd701, 0x17c0, 0x1680, 0xd641, 
    0xd201, 0x12c0, 0x1380, 0xd341, 0x1100, 0xd1c1, 0xd081, 0x1040, 
    0xf001, 0x30c0, 0x3180, 0xf141, 0x3300, 0xf3c1, 0xf281, 0x3240, 
    0x3600, 0xf6c1, 0xf781, 0x3740, 0xf501, 0x35c0, 0x3480, 0xf441, 
    0x3c00, 0xfcc1, 0xfd81, 0x3d40, 0xff01, 0x3fc0, 0x3e80, 0xfe41, 
    0xfa01, 0x3ac0, 0x3b80, 0xfb41, 0x3900, 0xf9c1, 0xf881, 0x3840, 
    0x2800, 0xe8c1, 0xe981, 0x2940, 0xeb01, 0x2bc0, 0x2a80, 0xea41, 
    0xee01, 0x2ec0, 0x2f80, 0xef41, 0x2d00, 0xedc1, 0xec81, 0x2c40, 
    0xe401, 0x24c0, 0x2580, 0xe541, 0x2700, 0xe7c1, 0xe681, 0x2640, 
    0x2200, 0xe2c1, 0xe381, 0x2340, 0xe101, 0x21c0, 0x2080, 0xe041, 
    0xa001, 0x60c0, 0x6180, 0xa141, 0x6300, 0xa3c1, 0xa281, 0x6240, 
    0x6600, 0xa6c1, 0xa781, 0x6740, 0xa501, 0x65c0, 0x6480, 0xa441, 
    0x6c00, 0xacc1, 0xad81, 0x6d40, 0xaf01, 0x6fc0, 0x6e80, 0xae41, 
    0xaa01, 0x6ac0, 0x6b80, 0xab41, 0x6900, 0xa9c1, 0xa881, 0x6840, 
    0x7800, 0xb8c1, 0xb981, 0x7940, 0xbb01, 0x7bc0, 0x7a80, 0xba41, 
    0xbe01, 0x7ec0, 0x7f80, 0xbf41, 0x7d00, 0xbdc1, 0xbc81, 0x7c40, 
    0xb401, 0x74c0, 0x7580, 0xb541, 0x7700, 0xb7c1, 0xb681, 0x7640, 
    0x7200, 0xb2c1, 0xb381, 0x7340, 0xb101, 0x71c0, 0x7080, 0xb041, 
    0x5000, 0x90c1, 0x9181, 0x5140, 0x9301, 0x53c0, 0x5280, 0x9241, 
    0x9601, 0x56c0, 0x5780, 0x9741, 0x5500, 0x95c1, 0x9481, 0x5440, 
    0x9c01, 0x5cc0, 0x5d80, 0x9d41, 0x5f00, 0x9fc1, 0x9e81, 0x5e40, 
    0x5a00, 0x9ac1, 0x9b81, 0x5b40, 0x9901, 0x59c0, 0x5880, 0x9841, 
    0x8801, 0x48c0, 0x4980, 0x8941, 0x4b00, 0x8bc1, 0x8a81, 0x4a40, 
    0x4e00, 0x8ec1, 0x8f81, 0x4f40, 0x8d01, 0x4dc0, 0x4c80, 0x8c41, 
    0x4400, 0x84c1, 0x8581, 0x4540, 0x8701, 0x47c0, 0x4680, 0x8641, 
    0x8201, 0x42c0, 0x4380, 0x8341, 0x4100, 0x81c1, 0x8081, 0x4040]

def crc16_modbus(init_crc, dat, len):
    crc = [init_crc >> 8, init_crc & 0xFF]    
    for b in dat:
        tmp = crc16_table[crc[0] ^ b]
        crc[0] = (tmp & 0xFF) ^ crc[1]
        crc[1] = tmp>>8
    
    return (crc[0]|crc[1]<<8)

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
        f"v4l2src device=/dev/video0 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false",
        f"v4l2src device=/dev/video1 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false",
        f"libcamerasrc ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false"
    ]
    return pipelines

class RTSPServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps, bitrate_kbps):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.appsrc = None
        
        # appsrc 기반 파이프라인 생성
        pipeline = (
            f"appsrc name=source is-live=true format=3 do-timestamp=true "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw ! "
            f"x264enc bitrate={bitrate_kbps} tune=zerolatency speed-preset=superfast key-int-max=30 ! "
            "rtph264pay name=pay0 pt=96"
        )
        self.set_launch(pipeline)
        self.set_shared(True)
        
        # media가 준비되면 appsrc를 가져오기 위한 시그널 연결
        self.connect("media-configure", self.on_media_configure)
    
    def on_media_configure(self, factory, media):
        """미디어가 설정될 때 appsrc 참조 가져오기"""
        element = media.get_element()
        if element:
            self.appsrc = element.get_by_name("source")
            if self.appsrc:
                print("appsrc configured successfully")

def setup_rtsp_server(port, mount_point, width, height, fps, bitrate_kbps):
    """RTSP 서버 설정 및 시작"""
    server = GstRtspServer.RTSPServer()
    server.set_service(str(port))
    
    factory = RTSPServerFactory(width, height, fps, bitrate_kbps)
    
    mounts = server.get_mount_points()
    mounts.add_factory(mount_point, factory)
    
    server.attach(None)
    
    print(f"RTSP Server started at rtsp://192.168.10.219:{port}{mount_point}")
    print(f"Connect using: rtsp://192.168.10.219:{port}{mount_point}")
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
                data, _ = udp_socket.recvfrom(10)
            except socket.timeout:
                continue
            except Exception:
                break

            # 패킷 길이 및 헤더 체크 (헤더 2바이트 + 데이터 6바이트 + CRC 2바이트 = 10바이트)
            if len(data) < 10 or data[0] != 0x55 or data[1] != 0x66:
                continue

            # CRC-16 체크섬 검증 (헤더부터 zoom_cmd까지 8바이트)
            received_crc = struct.unpack('<H', data[8:10])[0]
            calculated_crc = crc16_modbus(0xFFFF, data[0:8], 8)
            
            if received_crc != calculated_crc:
                print(f"CRC mismatch: received={hex(received_crc)}, calculated={hex(calculated_crc)}")
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
            elif data[6] == 0x00 and target_selected:
                target_selected = False
                tracking = False
                feature_lock_active = False

            if zoom_cmd == 0x02 and zoom_command != 'zoom_in':
                zoom_command = 'zoom_in'
                zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x01 and zoom_command != 'zoom_out':
                zoom_command = 'zoom_out'
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
        print("GStreamer initialized successfully")
    except Exception as e:
        print(f"GStreamer initialization failed: {e}")

    serial_port = setup_serial()

    # GLib MainLoop를 별도 스레드에서 실행
    mainloop = GLib.MainLoop()
    mainloop_thread = threading.Thread(target=mainloop.run, daemon=True)
    mainloop_thread.start()

    # RTSP 서버 시작
    rtsp_server, rtsp_factory = setup_rtsp_server(
        RTSP_PORT, RTSP_MOUNT_POINT, 
        FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, BITRATE_KBPS
    )
    
    # UDP 수신 스레드 시작
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()

    cap_pipelines = gstreamer_camera_pipeline(FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS)
    cap = None
    
    for i, pipeline in enumerate(cap_pipelines):
        print(f"Trying camera pipeline {i+1}: {pipeline[:50]}...")
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"Successfully opened camera with pipeline {i+1}")
                break
            else:
                cap.release()
                cap = None
        except Exception as e:
            print(f"Pipeline {i+1} failed: {e}")
            if cap:
                cap.release()
                cap = None
    
    if cap is None:
        print("Trying direct OpenCV camera access...")
        for device_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"Successfully opened camera device {device_id} with direct OpenCV (native resolution)")
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
                print(f"Device {device_id} failed: {e}")
                if cap:
                    cap.release()
                    cap = None
    
    if cap is None:
        raise RuntimeError("Failed to open camera with any method. Check camera connections and permissions.")

    # appsrc는 RTSP factory에서 관리
    appsrc = None
    
    print("Waiting for RTSP client connection to start streaming...")
    print(f"Connect to: rtsp://192.168.10.219:{RTSP_PORT}{RTSP_MOUNT_POINT}")

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
            if frame_counter % (TARGET_FPS * 4) == 0:
                print(f"[{time.strftime('%H:%M:%S')}] frames: {frame_counter} tracking={tracking}")

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
            
            # LK 프레임 스케일 표시 (디버깅용)
            if LK_FRAME_SCALE < 1.0:
                cv2.putText(display_frame, f"LK Scale: {LK_FRAME_SCALE:.1f}", (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # RTSP 스트리밍 - factory의 appsrc가 준비되면 프레임 푸시
            if rtsp_factory.appsrc is not None:
                try:
                    buf = Gst.Buffer.new_allocate(None, display_frame.nbytes, None)
                    buf.fill(0, display_frame.tobytes())
                    ret = rtsp_factory.appsrc.emit("push-buffer", buf)
                    if ret != Gst.FlowReturn.OK:
                        # 클라이언트 연결 해제 등의 상황
                        pass
                except Exception as e:
                    # 에러 발생 시 무시 (클라이언트가 아직 연결 안됨 등)
                    pass

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
        print("Cleanup completed")

if __name__ == "__main__":
    main()
    