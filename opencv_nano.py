import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
import os
import math

# NCNN import 추가
try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    print("Warning: NCNN not available, falling back to OpenCV tracker")
    NCNN_AVAILABLE = False

os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

class NanoTrack:
    """NanoTrack 딥러닝 기반 트래커"""
    
    def __init__(self, backbone_param_path="./models/nanotrack_backbone_sim.param",
                 backbone_bin_path="./models/nanotrack_backbone_sim.bin",
                 head_param_path="./models/nanotrack_head_sim.param",
                 head_bin_path="./models/nanotrack_head_sim.bin"):
        
        # Configuration
        self.cfg = {
            'context_amount': 0.5,
            'exemplar_size': 127,
            'instance_size': 255,
            'score_size': 16,
            'total_stride': 16,
            'window_influence': 0.42,
            'penalty_k': 0.04,
            'lr': 0.34
        }
        
        # NCNN 모델 로드
        self.net_backbone = ncnn.Net()
        self.net_backbone.opt.num_threads = 1
        self.net_backbone.opt.use_local_pool_allocator = True
        self.net_backbone.opt.use_vulkan_compute = False
        self.net_backbone.load_param(backbone_param_path)
        self.net_backbone.load_model(backbone_bin_path)
        
        self.net_head = ncnn.Net()
        self.net_head.opt.num_threads = 1
        self.net_head.opt.use_local_pool_allocator = True
        self.net_head.opt.use_vulkan_compute = False
        self.net_head.load_param(head_param_path)
        self.net_head.load_model(head_bin_path)
        
        # 상태 변수
        self.center = None
        self.target_sz = None
        self.bbox = None
        self.zf = None
        self.channel_ave = None
        self.im_h = 0
        self.im_w = 0
        
        # 실패 추적용
        self.failed_frames = 0
        self.max_failed_frames = 10
        self.confidence_threshold = 0.2
        
        # 윈도우와 그리드 생성
        self._create_window()
        self._create_grids()
        
    def _create_window(self):
        """코사인 윈도우 생성"""
        score_size = self.cfg['score_size']
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        
    def _create_grids(self):
        """검색 그리드 생성"""
        sz = self.cfg['score_size']
        x, y = np.meshgrid(np.arange(sz), np.arange(sz))
        self.grid_to_search_x = x.flatten() * self.cfg['total_stride']
        self.grid_to_search_y = y.flatten() * self.cfg['total_stride']
        
    def _get_subwindow_tracking(self, im, pos, model_sz, original_sz, avg_chans):
        """이미지에서 서브윈도우 추출"""
        c = (original_sz + 1) / 2
        context_xmin = round(pos[0] - c)
        context_xmax = context_xmin + original_sz - 1
        context_ymin = round(pos[1] - c)
        context_ymax = context_ymin + original_sz - 1
        
        left_pad = max(0, -context_xmin)
        top_pad = max(0, -context_ymin)
        right_pad = max(0, context_xmax - im.shape[1] + 1)
        bottom_pad = max(0, context_ymax - im.shape[0] + 1)
        
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad
        
        if top_pad > 0 or left_pad > 0 or right_pad > 0 or bottom_pad > 0:
            te_im = cv2.copyMakeBorder(im, top_pad, bottom_pad, left_pad, right_pad,
                                       cv2.BORDER_CONSTANT, value=avg_chans)
            im_patch = te_im[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1]
        else:
            im_patch = im[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1]
            
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        return im_patch
    
    def init_with_bbox(self, frame, bbox):
        """OpenCV 트래커와 호환되는 초기화 (bbox 입력)"""
        x, y, w, h = bbox
        center_point = (x + w//2, y + h//2)
        return self.init(frame, center_point, bbox)
    
    def init(self, frame, point, bbox=None):
        """트래커 초기화"""
        if bbox is None:
            # 기존 ROI 선택 로직
            x, y = int(point[0]), int(point[1])
            roi_size = 100
            half_size = roi_size // 2
            
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(frame.shape[1], x + half_size)
            y2 = min(frame.shape[0], y + half_size)
            
            self.bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            self.bbox = bbox
            
        x, y, w, h = self.bbox
        
        # NanoTrack 초기화
        self.center = [x + w//2, y + h//2]
        self.target_sz = [w, h]
        
        # 이미지 정보 저장
        self.im_h = frame.shape[0]
        self.im_w = frame.shape[1]
        self.channel_ave = cv2.mean(frame)[:3]
        
        # 템플릿 특징 추출
        wc_z = self.target_sz[0] + self.cfg['context_amount'] * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.cfg['context_amount'] * sum(self.target_sz)
        s_z = round(math.sqrt(wc_z * hc_z))
        
        z_crop = self._get_subwindow_tracking(frame, self.center, 
                                             self.cfg['exemplar_size'], 
                                             int(s_z), self.channel_ave)
        
        # NCNN으로 특징 추출
        ex = self.net_backbone.create_extractor()
        
        z_crop_rgb = cv2.cvtColor(z_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(z_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 
                                      z_crop.shape[1], z_crop.shape[0])
        
        ex.input("input", mat_in)
        _, self.zf = ex.extract("output")
        
        self.failed_frames = 0
        
        print(f"NanoTrack initialized at {self.center} with bbox {self.bbox}")
        return True
        
    def update(self, frame):
        """트래킹 업데이트 (OpenCV 트래커와 동일한 인터페이스)"""
        if self.center is None:
            return False, self.bbox
            
        # 검색 영역 계산
        wc_z = self.target_sz[0] + self.cfg['context_amount'] * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.cfg['context_amount'] * sum(self.target_sz)
        s_z = math.sqrt(wc_z * hc_z)
        scale_z = self.cfg['exemplar_size'] / s_z
        
        d_search = (self.cfg['instance_size'] - self.cfg['exemplar_size']) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        
        # 검색 영역 추출
        x_crop = self._get_subwindow_tracking(frame, self.center,
                                             self.cfg['instance_size'],
                                             int(s_x), self.channel_ave)
        
        # 특징 추출 (backbone)
        ex_backbone = self.net_backbone.create_extractor()
        
        x_crop_rgb = cv2.cvtColor(x_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(x_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                      x_crop.shape[1], x_crop.shape[0])
        
        ex_backbone.input("input", mat_in)
        _, xf = ex_backbone.extract("output")
        
        # Head 네트워크로 예측
        ex_head = self.net_head.create_extractor()
        
        ex_head.input("input1", self.zf)
        ex_head.input("input2", xf)
        
        _, cls_score = ex_head.extract("output1")
        _, bbox_pred = ex_head.extract("output2")
        
        # 점수맵 처리
        score_size = self.cfg['score_size']
        cls_score_np = np.array(cls_score)[1, :, :].flatten()
        cls_score_sigmoid = 1 / (1 + np.exp(-cls_score_np))
        
        # Bounding box regression
        bbox_pred_np = np.array(bbox_pred).reshape(4, -1)
        
        # 예측 좌표 계산
        pred_x1 = self.grid_to_search_x - bbox_pred_np[0]
        pred_y1 = self.grid_to_search_y - bbox_pred_np[1]
        pred_x2 = self.grid_to_search_x + bbox_pred_np[2]
        pred_y2 = self.grid_to_search_y + bbox_pred_np[3]
        
        # 크기 페널티 계산
        w = pred_x2 - pred_x1
        h = pred_y2 - pred_y1
        
        # 페널티 적용
        target_sz_prod = math.sqrt((self.target_sz[0] + sum(self.target_sz) * 0.5) * 
                                  (self.target_sz[1] + sum(self.target_sz) * 0.5))
        
        s_c = np.maximum(w / target_sz_prod, target_sz_prod / w) * \
              np.maximum(h / target_sz_prod, target_sz_prod / h)
        r_c = np.maximum((self.target_sz[0] / self.target_sz[1]) / (w / (h + 1e-6)),
                        (w / (h + 1e-6)) / (self.target_sz[0] / self.target_sz[1]))
        
        penalty = np.exp(-(s_c * r_c - 1) * self.cfg['penalty_k'])
        pscore = penalty * cls_score_sigmoid * (1 - self.cfg['window_influence']) + \
                self.window * self.cfg['window_influence']
        
        # 최대 점수 위치
        best_idx = np.argmax(pscore)
        best_score = cls_score_sigmoid[best_idx]
        
        # 신뢰도 체크
        if best_score < self.confidence_threshold:
            self.failed_frames += 1
            if self.failed_frames > self.max_failed_frames:
                print(f"Tracking failed: confidence={best_score:.3f}")
                return False, self.bbox
            return True, self.bbox
        
        self.failed_frames = 0
        
        # 위치 업데이트
        pred_xs = (pred_x1[best_idx] + pred_x2[best_idx]) / 2
        pred_ys = (pred_y1[best_idx] + pred_y2[best_idx]) / 2
        pred_w = pred_x2[best_idx] - pred_x1[best_idx]
        pred_h = pred_y2[best_idx] - pred_y1[best_idx]
        
        diff_xs = (pred_xs - self.cfg['instance_size'] / 2) / scale_z
        diff_ys = (pred_ys - self.cfg['instance_size'] / 2) / scale_z
        
        # Learning rate
        lr = penalty[best_idx] * best_score * self.cfg['lr']
        
        # 중심점 업데이트
        self.center[0] += diff_xs
        self.center[1] += diff_ys
        
        # 크기 업데이트
        self.target_sz[0] = self.target_sz[0] * (1 - lr) + pred_w / scale_z * lr
        self.target_sz[1] = self.target_sz[1] * (1 - lr) + pred_h / scale_z * lr
        
        # 경계 체크
        self.center[0] = np.clip(self.center[0], 0, self.im_w)
        self.center[1] = np.clip(self.center[1], 0, self.im_h)
        self.target_sz[0] = np.clip(self.target_sz[0], 10, self.im_w)
        self.target_sz[1] = np.clip(self.target_sz[1], 10, self.im_h)
        
        # 바운딩 박스 계산
        self.bbox = (
            int(self.center[0] - self.target_sz[0] / 2),
            int(self.center[1] - self.target_sz[1] / 2),
            int(self.target_sz[0]),
            int(self.target_sz[1])
        )
        
        return True, self.bbox


# 전역 변수들
current_frame = None
roi = None
tracker = None
tracking = False
latest_point = None
new_point_received = False
target_selected = False
serial_port = None
zoom_level = 1.0
zoom_command = None
zoom_center = None
center_x = 0
center_y = 0
prev_center_x = 0
prev_center_y = 0
velocity_x = 0
velocity_y = 0
tracker_fail_count = 0

# NanoTrack 모델 (전역으로 한 번만 로드)
nanotrack_model = None
USE_NANOTRACK = NCNN_AVAILABLE

def camera_init(capture, resolution_index=0):
    if capture.isOpened():
        pass
    else:
        exit()
        return False
    
    resolutions = [
        (1920, 1080),
        (3840, 2160),
        (4208, 3120)
    ]

    if resolution_index < 0 or resolution_index >= len(resolutions):
        return False
        
    frame_width, frame_height = resolutions[resolution_index]

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FPS, 25)

    return True

def setup_serial():
    try:
        ser = serial.Serial(
            port='/dev/ttyTHS1',
            baudrate=57600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        return ser
    except Exception as e:
        return None

def send_data_to_serial(x, y, is_tracking):
    global serial_port
    
    x = int((-1 + x * (2/1280)) * 1000)
    y = int((1 - y * (2/720)) * 1000)
    
    if serial_port is None or not serial_port.is_open:
        return
    
    try:
        data = struct.pack('<BBhhB', 
                          0xBB, 0x88,
                          x, y,
                          0xFF if is_tracking else 0x00)
        checksum = 0
        for byte in data:
            checksum ^= byte
        data += bytes([checksum])
        serial_port.write(data)
    except Exception as e:
        try:
            if serial_port and serial_port.is_open:
                serial_port.close()
            time.sleep(1)
            serial_port = setup_serial()
        except:
            pass

def udp_receiver():
    global latest_point, new_point_received, target_selected, zoom_command, tracking, zoom_center
    
    prev_x = None
    prev_y = None
    prev_target_selected = None
    prev_zoom_cmd = None
    
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.bind(('192.168.10.219', 5001))
            break
        except Exception as e:
            retry_count += 1
            time.sleep(5)
    
    if retry_count == max_retries:
        return
    
    try:
        while True:
            data, addr = udp_socket.recvfrom(8)
            if len(data) < 8:
                continue
            if data[0] != 0xAA or data[1] != 0x77:
                continue
            
            x = struct.unpack('<H', data[2:4])[0]
            y = struct.unpack('<H', data[4:6])[0]
            is_target_selected = data[6] == 0xFF
            zoom_cmd = data[7]
            
            if (x == prev_x and y == prev_y and 
                is_target_selected == prev_target_selected and 
                zoom_cmd == prev_zoom_cmd):
                continue
            
            prev_x, prev_y = x, y
            prev_target_selected = is_target_selected
            prev_zoom_cmd = zoom_cmd
            
            orig_x, orig_y = x, y
            
            if 'display_to_original_coord' in globals():
                try:
                    orig_x, orig_y = display_to_original_coord(x, y)
                except:
                    pass
            
            if is_target_selected:
                latest_point = (orig_x, orig_y)
                new_point_received = True
                target_selected = True
            elif data[6] == 0x00 and target_selected:
                target_selected = False
                tracking = False
            
            if zoom_cmd == 0x02 and zoom_command != 'zoom_in':
                zoom_command = 'zoom_in'
                zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x01 and zoom_command != 'zoom_out':
                zoom_command = 'zoom_out'
                zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x00:
                zoom_command = None
                
    except Exception as e:
        pass
    finally:
        udp_socket.close()

def process_new_coordinate(frame):
    global latest_point, new_point_received, tracker, tracking, roi, current_frame
    global zoom_level, center_x, center_y, prev_center_x, prev_center_y
    global velocity_x, velocity_y, tracker_fail_count, USE_NANOTRACK, nanotrack_model
    
    if new_point_received:
        new_point_received = False
        x, y = latest_point
        current_frame = frame.copy()
        
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            roi_size = int(100 / zoom_level)
            left = max(0, x - roi_size // 2)
            top = max(0, y - roi_size // 2)
            right = min(frame.shape[1], x + roi_size // 2)
            bottom = min(frame.shape[0], y + roi_size // 2)
            roi = (left, top, right - left, bottom - top)
            
            center_x = left + (right - left) / 2
            center_y = top + (bottom - top) / 2
            
            # 초기화 시 속도와 이전 위치 리셋
            prev_center_x = center_x
            prev_center_y = center_y
            velocity_x = 0
            velocity_y = 0
            tracker_fail_count = 0
            
            # NanoTrack 또는 OpenCV 트래커 선택
            if USE_NANOTRACK and nanotrack_model is not None:
                tracker = nanotrack_model
                tracker.init_with_bbox(frame, roi)
                print("Using NanoTrack for tracking")
            else:
                # Fallback to OpenCV tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, roi)
                print("Using OpenCV CSRT for tracking")
            
            tracking = True
        else:
            pass

def main():
    global current_frame, tracker, tracking, roi, target_selected, serial_port
    global zoom_level, zoom_command, zoom_center, nanotrack_model, USE_NANOTRACK
    
    # NanoTrack 모델 로드 시도
    if USE_NANOTRACK:
        try:
            print("Loading NanoTrack models...")
            nanotrack_model = NanoTrack(
                backbone_param_path="./models/nanotrack_backbone_sim.param",
                backbone_bin_path="./models/nanotrack_backbone_sim.bin",
                head_param_path="./models/nanotrack_head_sim.param",
                head_bin_path="./models/nanotrack_head_sim.bin"
            )
            print("NanoTrack models loaded successfully!")
        except Exception as e:
            print(f"Failed to load NanoTrack models: {e}")
            print("Falling back to OpenCV CSRT tracker")
            USE_NANOTRACK = False
            nanotrack_model = None
    
    serial_port = setup_serial()
    
    global display_to_original_coord
    
    def display_to_original_coord(disp_x, disp_y, cur_zoom_level=1.0, cur_zoom_x1=0, cur_zoom_y1=0):
        if cur_zoom_level <= 1.0:
            return int(disp_x), int(disp_y)
        
        rel_x = disp_x / cur_zoom_level
        rel_y = disp_y / cur_zoom_level
        
        orig_x = int(rel_x + cur_zoom_x1)
        orig_y = int(rel_y + cur_zoom_y1)
        
        orig_x = max(0, min(orig_x, 1280-1))
        orig_y = max(0, min(orig_y, 720-1))
        
        return orig_x, orig_y
    
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()
    
    cap = cv2.VideoCapture(0)
    
    if not camera_init(cap):
        exit()
    
    pipeline_str = (
        "appsrc name=source is-live=true format=3 do-timestamp=true ! "
        "video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! video/x-raw ! "
        "x264enc bitrate=6000 tune=zerolatency speed-preset=superfast key-int-max=1 ! "
        "h264parse ! "
        "rtph264pay config-interval=1 ! "
        "queue max-size-buffers=400 max-size-time=0 max-size-bytes=0 ! "
        "udpsink host=192.168.10.204 port=10010 buffer-size=2097152 sync=true async=false"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("source")
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    def on_bus_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
    bus.connect("message", on_bus_message)
    
    pipeline.set_state(Gst.State.PLAYING)
    
    original_frame = None
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            frame_counter += 1
            
            if not ret:
                time.sleep(0.1)
                continue
            
            original_frame = cv2.resize(frame, (1280, 720))
            frame = original_frame.copy()
            
            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level += 0.5
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level -= 0.5
                zoom_command = None
            
            current_frame = original_frame.copy()
            
            if frame_counter % 3 == 0:
                process_new_coordinate(original_frame)
            
            if not target_selected:
                tracking = False
                center_x = 0
                center_y = 0
                prev_center_x = 0 
                prev_center_y = 0
                velocity_x = 0
                velocity_y = 0

            if tracking and tracker is not None:
                if frame_counter % 3 == 0:
                    success, bbox = tracker.update(original_frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        roi = (x, y, w, h)
                        
                        # 이전 중심점 저장
                        prev_center_x = center_x
                        prev_center_y = center_y
                        
                        # 새로운 중심점 계산
                        center_x = x + w / 2
                        center_y = y + h / 2
                        
                        # 속도 계산 (3프레임 동안의 이동)
                        if prev_center_x != 0 and prev_center_y != 0:
                            velocity_x = (center_x - prev_center_x) / 3
                            velocity_y = (center_y - prev_center_y) / 3
                        else:
                            velocity_x = 0
                            velocity_y = 0
                        
                        tracker_fail_count = 0
                    else:
                        # 트래커 실패 시 예측된 위치로 이동
                        center_x += velocity_x * 3
                        center_y += velocity_y * 3
                        tracker_fail_count += 1
                        
                        # 5번 이상 실패하면 트래킹 중지
                        if tracker_fail_count > 5:
                            tracking = False
                else:
                    # 중간 프레임에서는 선형 보간으로 위치 예측
                    center_x += velocity_x
                    center_y += velocity_y
            
            display_frame = original_frame.copy()
            
            zoom_x1, zoom_y1 = 0, 0
            zoom_applied = False
            
            if zoom_level > 1.0 and zoom_center is not None:
                h, w = display_frame.shape[:2]
                
                center_x_zoom, center_y_zoom = zoom_center
                
                zoom_width = int(w / zoom_level)
                zoom_height = int(h / zoom_level)
                
                zoom_x1 = max(0, center_x_zoom - zoom_width // 2)
                zoom_y1 = max(0, center_y_zoom - zoom_height // 2)
                
                if zoom_x1 + zoom_width > w:
                    zoom_x1 = w - zoom_width
                if zoom_y1 + zoom_height > h:
                    zoom_y1 = h - zoom_height
                
                roi_zoom = display_frame[zoom_y1:zoom_y1+zoom_height, zoom_x1:zoom_x1+zoom_width]
                display_frame = cv2.resize(roi_zoom, (w, h))
                zoom_applied = True
            
            def original_to_display_coord(orig_x, orig_y):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(orig_x), int(orig_y)
                
                rel_x = orig_x - zoom_x1
                rel_y = orig_y - zoom_y1
                
                disp_x = int(rel_x * zoom_level)
                disp_y = int(rel_y * zoom_level)
                
                h, w = display_frame.shape[:2]
                disp_x = max(0, min(disp_x, w-1))
                disp_y = max(0, min(disp_y, h-1))
                
                return disp_x, disp_y
                
            def local_display_to_original_coord(disp_x, disp_y):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(disp_x), int(disp_y)
                
                rel_x = disp_x / zoom_level
                rel_y = disp_y / zoom_level
                
                orig_x = int(rel_x + zoom_x1)
                orig_y = int(rel_y + zoom_y1)
                
                h, w = original_frame.shape[:2]
                orig_x = max(0, min(orig_x, w-1))
                orig_y = max(0, min(orig_y, h-1))
                
                return orig_x, orig_y
                
            display_to_original_coord = lambda dx, dy: local_display_to_original_coord(dx, dy)
            
            if tracking and tracker is not None and roi is not None:
                x, y, w, h = roi
                
                disp_x, disp_y = original_to_display_coord(x, y)
                disp_x2, disp_y2 = original_to_display_coord(x + w, y + h)
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y
                
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                
                disp_center_x, disp_center_y = original_to_display_coord(int(center_x), int(center_y))
                cv2.circle(display_frame, (disp_center_x, disp_center_y), 5, (0, 0, 255), -1)
                
                send_data_to_serial(center_x, center_y, tracking)
            
            # 트래커 타입 표시
            tracker_type = "NanoTrack" if (USE_NANOTRACK and isinstance(tracker, NanoTrack)) else "CSRT"
            status = f"{tracker_type}: X={int(center_x)}, Y={int(center_y)}" if tracking else f"{tracker_type}: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking else (0, 0, 255), 2)
            
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            buffer = Gst.Buffer.new_allocate(None, display_frame.nbytes, None)
            buffer.fill(0, display_frame.tobytes())
            buffer.pts = Gst.CLOCK_TIME_NONE
            buffer.dts = Gst.CLOCK_TIME_NONE
            appsrc.emit("push-buffer", buffer)

            try:
                if open("stop.signal", "r").close() or False:
                    break
            except:
                pass

    except KeyboardInterrupt:
        pass
    
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if serial_port and serial_port.is_open:
            serial_port.close()
        pipeline.set_state(Gst.State.NULL)
        cap.release()

if __name__ == "__main__":
    main()