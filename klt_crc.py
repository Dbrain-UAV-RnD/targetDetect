import os

# 환경 변수 설정
gst_plugin_paths = [
    '/usr/lib/aarch64-linux-gnu/gstreamer-1.0',
]
existing_gst_path = os.environ.get('GST_PLUGIN_PATH', '')
os.environ['GST_PLUGIN_PATH'] = ':'.join(gst_plugin_paths + [existing_gst_path] if existing_gst_path else gst_plugin_paths)

gi_typelib_paths = [
    '/usr/lib/aarch64-linux-gnu/girepository-1.0',
    '/usr/lib/girepository-1.0',
]
existing_gi_path = os.environ.get('GI_TYPELIB_PATH', '')
os.environ['GI_TYPELIB_PATH'] = ':'.join(gi_typelib_paths + [existing_gi_path] if existing_gi_path else gi_typelib_paths)

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

TCP_HOST = '192.168.144.61'
TCP_PORT = 37260

RTSP_PORT = 8554
RTSP_MOUNT_POINT = '/video0'
BITRATE_KBPS = 6000
##
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
prev_roi_center = None  # 이전 ROI 중심점 저장용
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
        self.last_debug_time = time.time()

        pipeline = (
            f"appsrc name=source is-live=true format=3 do-timestamp=true "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={bitrate_kbps} tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 "
            f"rc-lookahead=0 sync-lookahead=0 sliced-threads=true ! "
            "rtph264pay name=pay0 pt=96 config-interval=1 aggregate-mode=zero-latency"
        )
        self.set_launch(pipeline)
        self.set_shared(True)

        self.connect("media-configure", self.on_media_configure)
        self.connect("media-constructed", self.on_media_constructed)
    
    def on_media_constructed(self, factory, media):
        pass

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
                appsrc.set_property('min-latency', 0)
                appsrc.set_property('max-latency', 0)
                appsrc.set_property('leaky-type', 2)

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

def tcp_receiver():
    global latest_point, new_point_received, target_selected, zoom_command, tracking, zoom_center, feature_lock_active
    global prev_roi_center

    prev_x = prev_y = None
    prev_target_selected = None
    prev_zoom_cmd = None

    max_retries, retry_delay = 30, 1.0
    
    # TCP 서버 소켓 생성
    server_socket = None
    for attempt in range(max_retries):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((TCP_HOST, TCP_PORT))
            server_socket.listen(1)  # 최대 1개의 대기 연결
            server_socket.settimeout(1.0)  # accept timeout
            print(f"TCP Server listening on {TCP_HOST}:{TCP_PORT}")
            break
        except Exception as e:
            print(f"TCP bind attempt {attempt+1} failed: {e}")
            if server_socket:
                try:
                    server_socket.close()
                except:
                    pass
                server_socket = None
            time.sleep(retry_delay)
    else:
        print("Failed to create TCP server")
        return

    def process_packet(data):
        nonlocal prev_x, prev_y, prev_target_selected, prev_zoom_cmd
        ###
        print(f"[Packet] Length: {len(data)}, Hex: {' '.join(f'{b:02X}' for b in data)}")
        
        # 최소 패킷 크기 체크: 2(header) + 1 + 2(len) + 2 + 1 + 1(최소 data) + 2(crc) = 11
        if len(data) < 11:
            return

        # 헤더 체크 (0x55 0x66으로 가정)
        if data[0] != 0x55 or data[1] != 0x66:
            return

        # 3번째 바이트가 1인지 체크
        if data[2] != 0x01:
            return

        # 데이터 길이 읽기 (리틀 엔디안)
        data_length = struct.unpack('<H', data[3:5])[0]

        # 4~5번째 바이트가 0인지 체크
        if data[5] != 0x00 or data[6] != 0x00:
            return

        # 전체 패킷 길이 계산: 2 + 1 + 2 + 2 + 1 + data_length + 2
        expected_length = 10 + data_length
        if len(data) < expected_length:
            return

        # 명령 바이트 (0 또는 4~6)
        cmd_byte = data[7]
        if cmd_byte not in [0x00, 0x04, 0x05, 0x06]:
            return

        # 가변 데이터 추출
        var_data = data[8:8+data_length]

        # CRC 체크
        crc_start = 8 + data_length
        if len(data) < crc_start + 2:
            return

        received_crc = struct.unpack('<H', data[crc_start:crc_start+2])[0]
        calculated_crc = crc16_modbus(0xFFFF, data[0:crc_start], crc_start)

        # if received_crc != calculated_crc:
        #     print("CRC Fail")
        #     return

        # 명령 바이트 해석
        # 0x00: TCP Heartbeat (가변 1바이트: 0x00 고정)
        # 0x04: AI Mode (가변 1바이트: 0 또는 1)
        # 0x05: Zoom Mode (가변 1바이트: -1/0/1)
        # 0x06: Target Selection (가변 9바이트: 1바이트 선택여부 + 4바이트 x + 4바이트 y)
        
        x = y = None
        is_target = False
        zoom_cmd = 0x00
        ai_mode = None
        
        if cmd_byte == 0x00:
            # TCP Heartbeat - 아무 동작 안함
            if data_length >= 1 and var_data[0] == 0x00:
                return
                
        elif cmd_byte == 0x04:
            # AI Mode
            if data_length >= 1:
                ai_mode = var_data[0]
                # AI 모드 활성화/비활성화 처리
                # 여기에 AI 모드 관련 로직 추가 가능
                
        elif cmd_byte == 0x05:
            # Zoom Mode
            if data_length >= 1:
                zoom_value = struct.unpack('b', var_data[0:1])[0]  # signed byte
                if zoom_value == 1:
                    zoom_cmd = 0x02  # zoom in
                elif zoom_value == -1:
                    zoom_cmd = 0x01  # zoom out
                else:
                    zoom_cmd = 0x00  # none
                    
        elif cmd_byte == 0x06:
            # Target Selection (9바이트: 1바이트 선택여부 + 2바이트 x1 + 2바이트 y1 + 2바이트 x2 + 2바이트 y2)
            if data_length >= 9:
                target_flag = var_data[0]
                x1 = struct.unpack('<H', var_data[1:3])[0]  # 좌상단 x
                y1 = struct.unpack('<H', var_data[3:5])[0]  # 좌상단 y
                x2 = struct.unpack('<H', var_data[5:7])[0]  # 우하단 x
                y2 = struct.unpack('<H', var_data[7:9])[0]  # 우하단 y
                
                # 중심점 계산
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                print(x1, y1, x2, y2, '->', x, y)
                is_target = (target_flag == 0x01)

        # 중복 데이터 필터링 (heartbeat 제외)
        if cmd_byte != 0x00:
            if (x == prev_x and y == prev_y and is_target == prev_target_selected and zoom_cmd == prev_zoom_cmd):
                return

            prev_x, prev_y = x, y
            prev_target_selected = is_target
            prev_zoom_cmd = zoom_cmd

        # Target Selection 처리 (cmd_byte == 0x06)
        if cmd_byte == 0x06 and x is not None and y is not None:
            orig_x, orig_y = x, y
            try:
                orig_x, orig_y = display_to_original_coord(x, y)
            except Exception:
                pass

            if is_target:
                latest_point = (orig_x, orig_y)
                new_point_received = True
                target_selected = True
                # 타겟 선택 시 줌 센터 업데이트
                if zoom_center is None:
                    zoom_center = (orig_x, orig_y)
            else:
                # 타겟 선택 해제
                target_selected = False
                tracking = False
                feature_lock_active = False
                prev_roi_center = None  # 이전 중심점 초기화
                print("[Tracking] 타겟 선택 해제 - 이전 중심점 초기화")

        # Zoom 명령 처리 (cmd_byte == 0x05)
        if cmd_byte == 0x05:
            if zoom_cmd == 0x02 and zoom_command != 'zoom_in':
                zoom_command = 'zoom_in'
                # zoom_center가 없으면 화면 중앙으로
                if zoom_center is None:
                    zoom_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
            elif zoom_cmd == 0x01 and zoom_command != 'zoom_out':
                zoom_command = 'zoom_out'
                if zoom_center is None:
                    zoom_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
            elif zoom_cmd == 0x00:
                zoom_command = None

    # TCP 서버 메인 루프
    try:
        while True:
            client_socket = None
            try:
                # 클라이언트 연결 대기
                print("Waiting for client connection...")
                try:
                    client_socket, client_addr = server_socket.accept()
                except socket.timeout:
                    continue
                
                client_socket.settimeout(0.5)  # recv timeout
                print(f"Client connected from {client_addr}")
                
                # 클라이언트와 통신
                buffer = b''
                while True:
                    try:
                        chunk = client_socket.recv(1024)
                        if not chunk:
                            # 연결 종료
                            print("Client disconnected")
                            break
                        
                        buffer += chunk
                        
                        # 패킷 처리
                        while len(buffer) >= 11:  # 최소 패킷 크기
                            # 헤더 찾기
                            header_idx = -1
                            for i in range(len(buffer) - 1):
                                if buffer[i] == 0x55 and buffer[i+1] == 0x66:
                                    header_idx = i
                                    break
                            
                            if header_idx == -1:
                                # 헤더를 찾지 못하면 버퍼의 마지막 바이트만 남기고 제거
                                buffer = buffer[-1:] if len(buffer) > 0 else b''
                                break
                            
                            # 헤더 이전 데이터 제거
                            if header_idx > 0:
                                buffer = buffer[header_idx:]
                            
                            # 패킷 길이 확인
                            if len(buffer) < 11:
                                break
                            
                            # 패킷 파싱
                            data = buffer
                            
                            # 처리 가능한지 확인
                            if data[2] != 0x01:
                                buffer = buffer[2:]  # 헤더 이후로 이동
                                continue
                            
                            try:
                                data_length = struct.unpack('<H', data[3:5])[0]
                            except:
                                buffer = buffer[2:]
                                continue
                            
                            if len(data) < 7 or data[5] != 0x00 or data[6] != 0x00:
                                buffer = buffer[2:]
                                continue
                            
                            expected_length = 10 + data_length
                            if len(buffer) < expected_length:
                                break  # 더 많은 데이터 필요
                            
                            # 완전한 패킷 추출
                            packet = buffer[:expected_length]
                            buffer = buffer[expected_length:]
                            
                            # 패킷 처리
                            process_packet(packet)
                            
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"Receive error: {e}")
                        break
                        
            except Exception as e:
                print(f"Connection error: {e}")
            finally:
                if client_socket:
                    try:
                        client_socket.close()
                    except:
                        pass
                time.sleep(0.1)  # 재연결 전 짧은 대기
                time.sleep(0.1)  # 재연결 전 짧은 대기
                
    finally:
        try:
            if server_socket:
                server_socket.close()
        except Exception:
            pass

def process_new_coordinate(gray_frame):
    global latest_point, new_point_received, tracking, prev_frame, tracking_points, roi_rect, roi_center
    global feature_lock_active, prev_roi_center

    if not new_point_received:
        return

    new_point_received = False
    x, y = latest_point

    if not (0 <= x < gray_frame.shape[1] and 0 <= y < gray_frame.shape[0]):
        return

    # 새로운 중심점이 이전 중심점과 같으면 무시
    if prev_roi_center is not None:
        prev_x, prev_y = prev_roi_center
        if x == prev_x and y == prev_y:
            print(f"[Tracking] 동일한 중심점 무시: ({x}, {y})")
            return
    
    # 새로운 중심점으로 업데이트
    prev_roi_center = (x, y)
    print(f"[Tracking] 새로운 타겟 중심점: ({x}, {y})")
    
    H, W = gray_frame.shape[:2]
    roi_rect = clamp_roi(x, y, ROI_SIZE, ROI_SIZE, W, H)
    roi_center = (x, y)
    
    tracking_points = detect_points_in_roi(gray_frame, roi_rect)
    
    if tracking_points is not None and len(tracking_points) > 0:
        prev_frame = gray_frame
        tracking = True
        feature_lock_active = LOCK_INITIAL_FEATURES
        print(f"[Tracking] 추적 시작 - 특징점 개수: {len(tracking_points)}")
    else:
        feature_lock_active = False
        print(f"[Tracking] 특징점을 찾을 수 없음")

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

    tcp_thread = threading.Thread(target=tcp_receiver, daemon=True)
    tcp_thread.start()

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
                    cap.set(cv2.CAP_PROP_FPS, 30)

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

            if rtsp_factory.appsrc_list:
                try:
                    buf = Gst.Buffer.new_allocate(None, display_frame.nbytes, None)
                    buf.fill(0, display_frame.tobytes())

                    for i, appsrc in enumerate(rtsp_factory.appsrc_list):
                        if appsrc is not None:
                            ret = appsrc.emit("push-buffer", buf)

                    rtsp_factory.frame_count += 1

                    now = time.time()
                    if now - rtsp_factory.last_debug_time >= 5.0:
                        rtsp_factory.last_debug_time = now

                except Exception as e:
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

if __name__ == "__main__":
    main()