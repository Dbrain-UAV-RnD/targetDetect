```python
import cv2
import socket
import threading
import struct
import time
import serial
import gi
import os
import numpy as np
import math

# NCNN import 추가
try:
    import ncnn
    NCNN_AVAILABLE = True
    print(f"NCNN version: {ncnn.__version__ if hasattr(ncnn, '__version__') else 'unknown'}")
except ImportError:
    print("Warning: NCNN not available, falling back to template matching")
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
                 head_bin_path="./models/nanotrack_head_sim.bin",
                 fixed_roi_size=100):  # 고정 ROI 크기 추가
        
        # Configuration (튜닝된 파라미터: window_influence 줄임, penalty_k 증가, lr 줄임)
        self.cfg = {
            'context_amount': 0.5,
            'exemplar_size': 127,
            'instance_size': 255,
            'score_size': 16,
            'total_stride': 16,
            'window_influence': 0.30,  # 줄여서 penalty에 더 의존하게 함 (jitter 감소)
            'penalty_k': 0.06,         # 증가시켜 크기 변화에 더 엄격하게 (tracking 안정성 향상)
            'lr': 0.28,                # 줄여서 위치 업데이트를 더 부드럽게 (jitter 감소)
            'fixed_roi_size': fixed_roi_size,  # 고정 ROI 크기
            'enable_size_update': False  # 크기 업데이트 비활성화
        }
        
        # NCNN 모델 로드 (스레드 설정을 load_param 전에)
        self.net_backbone = ncnn.Net()
        try:
            # 버전에 따라 사용 가능한 옵션 설정
            self.net_backbone.opt.num_threads = 1
            if hasattr(self.net_backbone.opt, 'use_vulkan_compute'):
                self.net_backbone.opt.use_vulkan_compute = False
        except:
            pass  # 옵션 설정 실패해도 기본값으로 진행
        self.net_backbone.load_param(backbone_param_path)
        self.net_backbone.load_model(backbone_bin_path)
        
        self.net_head = ncnn.Net()
        try:
            # 버전에 따라 사용 가능한 옵션 설정
            self.net_head.opt.num_threads = 1
            if hasattr(self.net_head.opt, 'use_vulkan_compute'):
                self.net_head.opt.use_vulkan_compute = False
        except:
            pass  # 옵션 설정 실패해도 기본값으로 진행
        self.net_head.load_param(head_param_path)
        self.net_head.load_model(head_bin_path)
        
        # 상태 변수
        self.center = None
        self.target_sz = None
        self.initial_sz = None  # 초기 크기 저장용
        self.bbox = None
        self.zf = None
        self.channel_ave = None
        self.im_h = 0
        self.im_w = 0
        
        # 실패 추적용 (RobustPitchTracker 호환성) - confidence_threshold 증가로 안정성 향상
        self.failed_frames = 0
        self.max_failed_frames = 10
        self.confidence_threshold = 0.25  # 증가시켜 낮은 신뢰도 예측 무시 (tracking 성능 향상)
        
        # 위치 스무딩을 위한 변수 추가 (jitter 감소)
        self.smooth_factor = 0.7  # 이전 위치와 새 위치를 블렌딩 (0.0: 완전 새 위치, 1.0: 이전 위치 유지)
        self.prev_center = None
        
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
    
    def simple_roi_selection(self, frame, point, roi_size=None):
        """ROI 선택 (고정 크기)"""
        if roi_size is None:
            roi_size = self.cfg['fixed_roi_size']
        
        x, y = int(point[0]), int(point[1])
        half_size = roi_size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # 경계에서도 ROI 크기 유지
        if x2 - x1 < roi_size:
            if x1 == 0:
                x2 = min(roi_size, frame.shape[1])
            else:
                x1 = max(0, x2 - roi_size)
        
        if y2 - y1 < roi_size:
            if y1 == 0:
                y2 = min(roi_size, frame.shape[0])
            else:
                y1 = max(0, y2 - roi_size)
        
        print(f"Fixed ROI: ({x1}, {y1}, {roi_size}, {roi_size}) at click point ({x}, {y})")
        return (x1, y1, roi_size, roi_size)
        
    def init(self, frame, point):
        """트래커 초기화 (고정 ROI 크기)"""
        self.bbox = self.simple_roi_selection(frame, point)
        x, y, w, h = self.bbox
        
        # NanoTrack 초기화
        self.center = [x + w//2, y + h//2]
        self.prev_center = self.center.copy()  # 스무딩 초기화
        self.target_sz = [w, h]
        self.initial_sz = [w, h]  # 초기 크기 저장 (고정용)
        
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
        # set_light_mode와 set_num_threads 제거 (이미 Net에서 설정됨)
        
        # BGR to RGB 변환 후 입력
        z_crop_rgb = cv2.cvtColor(z_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(z_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 
                                      z_crop.shape[1], z_crop.shape[0])
        
        ex.input("input", mat_in)
        _, self.zf = ex.extract("output")
        
        self.failed_frames = 0
        
        print(f"NanoTrack initialized with fixed ROI size: {w}x{h}")
        return True
        
    def update(self, frame):
        """트래킹 업데이트 (RobustPitchTracker와 동일한 인터페이스)"""
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
        # set_light_mode와 set_num_threads 제거 (이미 Net에서 설정됨)
        
        x_crop_rgb = cv2.cvtColor(x_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(x_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                      x_crop.shape[1], x_crop.shape[0])
        
        ex_backbone.input("input", mat_in)
        _, xf = ex_backbone.extract("output")
        
        # Head 네트워크로 예측
        ex_head = self.net_head.create_extractor()
        # set_light_mode와 set_num_threads 제거 (이미 Net에서 설정됨)
        
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
        
        # 중심점 업데이트 (스무딩 적용)
        new_center_x = self.prev_center[0] + diff_xs * (1 - self.smooth_factor) + self.prev_center[0] * self.smooth_factor
        new_center_y = self.prev_center[1] + diff_ys * (1 - self.smooth_factor) + self.prev_center[1] * self.smooth_factor
        self.center[0] = new_center_x
        self.center[1] = new_center_y
        self.prev_center = self.center.copy()
        
        # 크기 업데이트 (옵션에 따라)
        if self.cfg['enable_size_update']:
            # 크기 업데이트 활성화 시
            self.target_sz[0] = self.target_sz[0] * (1 - lr) + pred_w / scale_z * lr
            self.target_sz[1] = self.target_sz[1] * (1 - lr) + pred_h / scale_z * lr
        else:
            # 크기 고정 (초기 크기 유지)
            self.target_sz = self.initial_sz.copy()
        
        # 경계 체크
        self.center[0] = np.clip(self.center[0], 0, self.im_w)
        self.center[1] = np.clip(self.center[1], 0, self.im_h)
        
        # 바운딩 박스 계산 (고정 크기)
        self.bbox = (
            int(self.center[0] - self.target_sz[0] / 2),
            int(self.center[1] - self.target_sz[1] / 2),
            int(self.target_sz[0]),
            int(self.target_sz[1])
        )
        
        return True, self.bbox


class RobustPitchTracker:
    """폴백용 템플릿 매칭 트래커 (NCNN이 없을 경우)"""
    
    def __init__(self, fixed_roi_size=100):
        self.template = None
        self.template_gray = None
        self.bbox = None
        self.center = None
        
        self.fixed_roi_size = fixed_roi_size  # 고정 ROI 크기
        self.initial_template_size = None  # 초기 템플릿 크기 저장
        
        self.search_scale = 3.5  # 증가시켜 검색 영역 확대 (tracking 성능 향상)
        self.confidence_threshold = 0.25  # 증가시켜 낮은 신뢰도 무시 (tracking 안정성 향상)
        self.template_update_rate = 0.03  # 줄여서 템플릿 업데이트를 더 천천히 (jitter 감소)
        
        self.max_movement_ratio = 0.5
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_decay = 0.85  # 증가시켜 속도 감쇠를 더 부드럽게 (jitter 감소)
        
        self.failed_frames = 0
        self.max_failed_frames = 10
        
        self.prev_center_y = 0
        self.pitch_down_threshold = 50
        
        # 위치 스무딩을 위한 변수 추가 (jitter 감소)
        self.smooth_factor = 0.7  # 이전 위치와 새 위치를 블렌딩
        self.prev_center = None
    
    def simple_roi_selection(self, frame, point, roi_size=None):
        """ROI 선택 (고정 크기)"""
        if roi_size is None:
            roi_size = self.fixed_roi_size
            
        x, y = int(point[0]), int(point[1])
        half_size = roi_size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # 경계에서도 ROI 크기 유지
        if x2 - x1 < roi_size:
            if x1 == 0:
                x2 = min(roi_size, frame.shape[1])
            else:
                x1 = max(0, x2 - roi_size)
        
        if y2 - y1 < roi_size:
            if y1 == 0:
                y2 = min(roi_size, frame.shape[0])
            else:
                y1 = max(0, y2 - roi_size)
        
        print(f"Fixed ROI: ({x1}, {y1}, {roi_size}, {roi_size}) at click point ({x}, {y})")
        return (x1, y1, roi_size, roi_size)
    
    def init(self, frame, point):
        """트래커 초기화 (고정 ROI 크기)"""
        self.bbox = self.simple_roi_selection(frame, point)
        x, y, w, h = self.bbox
        
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.initial_template_size = (w, h)  # 초기 크기 저장
        
        self.center = [x + w//2, y + h//2]
        self.prev_center = self.center.copy()  # 스무딩 초기화
        self.prev_center_y = self.center[1]
        
        self.velocity_x = 0
        self.velocity_y = 0
        self.failed_frames = 0
        
        print(f"Template tracker initialized with fixed ROI size: {w}x{h}")
        return True
    
    def update(self, frame):
        if self.template is None or self.template_gray is None:
            return False, self.bbox
        
        predicted_x = self.center[0] + self.velocity_x
        predicted_y = self.center[1] + self.velocity_y
        
        # 고정 템플릿 크기 사용
        template_w, template_h = self.initial_template_size
        search_w = int(template_w * self.search_scale)
        search_h = int(template_h * self.search_scale)
        
        search_x1 = max(0, int(predicted_x - search_w // 2))
        search_y1 = max(0, int(predicted_y - search_h // 4))
        search_x2 = min(frame.shape[1], search_x1 + search_w)
        search_y2 = min(frame.shape[0], search_y1 + int(search_h * 1.5))
        
        search_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            self.failed_frames += 1
            return self.failed_frames <= self.max_failed_frames, self.bbox
        
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # 템플릿은 고정 크기로 매칭 (스케일 변화 없음)
        result = cv2.matchTemplate(search_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, best_score, _, best_loc = cv2.minMaxLoc(result)
        
        if best_score < self.confidence_threshold:
            self.failed_frames += 1
            
            if self.failed_frames <= self.max_failed_frames:
                if self.velocity_y > 0:
                    self.velocity_y *= 1.2
                
                self.center[0] += self.velocity_x
                self.center[1] += self.velocity_y
                
                self.center[0] = max(template_w//2, min(self.center[0], frame.shape[1] - template_w//2))
                self.center[1] = max(template_h//2, min(self.center[1], frame.shape[0] - template_h//2))
                
                # 고정 크기 bbox
                self.bbox = (
                    int(self.center[0] - template_w//2),
                    int(self.center[1] - template_h//2),
                    template_w,
                    template_h
                )
                
                return True, self.bbox
            else:
                print(f"Tracking failed: confidence={best_score:.3f}")
                return False, self.bbox
        
        self.failed_frames = 0
        
        # 새 중심점 계산
        new_center_x = search_x1 + best_loc[0] + template_w // 2
        new_center_y = search_y1 + best_loc[1] + template_h // 2
        
        # 스무딩 적용
        new_center_x = new_center_x * (1 - self.smooth_factor) + self.prev_center[0] * self.smooth_factor
        new_center_y = new_center_y * (1 - self.smooth_factor) + self.prev_center[1] * self.smooth_factor
        
        self.velocity_x = new_center_x - self.center[0]
        self.velocity_y = new_center_y - self.center[1]
        
        y_movement = new_center_y - self.prev_center_y
        if y_movement > self.pitch_down_threshold:
            print(f"Pitch down detected: {y_movement} pixels")
            self.confidence_threshold = 0.2
            self.search_scale = 4.0
        else:
            self.confidence_threshold = 0.3
            self.search_scale = 3.0
        
        self.prev_center_y = new_center_y
        
        self.center[0] = new_center_x
        self.center[1] = new_center_y
        self.prev_center = self.center.copy()
        
        self.velocity_x *= self.velocity_decay
        self.velocity_y *= self.velocity_decay
        
        # 고정 크기 bbox
        self.bbox = (
            int(self.center[0] - template_w//2),
            int(self.center[1] - template_h//2),
            template_w,
            template_h
        )
        
        # 템플릿 업데이트 (크기는 고정)
        if best_score > 0.6:
            x, y, w, h = self.bbox
            if 0 <= x < frame.shape[1] - w and 0 <= y < frame.shape[0] - h:
                new_template = frame[y:y+h, x:x+w]
                if new_template.size > 0 and new_template.shape == self.template.shape:
                    self.template = cv2.addWeighted(
                        self.template, 1 - self.template_update_rate,
                        new_template,
                        self.template_update_rate, 0
                    )
                    self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        return True, self.bbox


# 전역 변수들
current_frame = None
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
bbox = None

# 트래커 설정
FIXED_ROI_SIZE = 100  # 고정 ROI 크기 (픽셀) - 필요시 조정 가능
USE_NANOTRACK = NCNN_AVAILABLE  # NCNN 사용 가능 여부에 따라 결정

# NanoTrack 모델 (전역으로 한 번만 로드)
nanotrack_model = None

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
                print(f"Target selected at: {latest_point}")
            elif data[6] == 0x00 and target_selected:
                target_selected = False
                tracking = False
                print("Target deselected")
            
            if zoom_cmd == 0x02 and zoom_command != 'zoom_in':
                zoom_command = 'zoom_in'
                zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x01 and zoom_command != 'zoom_out':
                zoom_command = 'zoom_out'
                zoom_center = (orig_x, orig_y)
            elif zoom_cmd == 0x00:
                zoom_command = None
                
    except Exception as e:
        print(f"UDP receiver error: {e}")
    finally:
        udp_socket.close()

def process_new_coordinate(frame):
    global latest_point, new_point_received, tracker, tracking, current_frame
    global center_x, center_y, bbox, USE_NANOTRACK, nanotrack_model, FIXED_ROI_SIZE
    
    if new_point_received:
        new_point_received = False
        point = latest_point
        current_frame = frame.copy()
        
        print(f"Processing new coordinate: {point}")
        
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            # NanoTrack 또는 템플릿 매칭 선택 (둘 다 고정 ROI 사용)
            if USE_NANOTRACK and nanotrack_model is not None:
                tracker = nanotrack_model
                print(f"Using NanoTrack with fixed ROI size: {FIXED_ROI_SIZE}x{FIXED_ROI_SIZE}")
            else:
                tracker = RobustPitchTracker(fixed_roi_size=FIXED_ROI_SIZE)
                print(f"Using Template Matching with fixed ROI size: {FIXED_ROI_SIZE}x{FIXED_ROI_SIZE}")
            
            success = tracker.init(frame, point)
            
            if success:
                tracking = True
                bbox = tracker.bbox
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                print(f"Tracker initialized successfully with fixed bbox: {bbox}")
            else:
                tracking = False
                print("Failed to initialize tracker")
        else:
            print(f"Point {point} is out of bounds")

def main():
    global current_frame, tracker, tracking, target_selected, serial_port
    global zoom_level, zoom_command, zoom_center, center_x, center_y, bbox
    global nanotrack_model, USE_NANOTRACK
    
    # NanoTrack 모델 로드 시도
    if USE_NANOTRACK:
        try:
            print("Loading NanoTrack models...")
            nanotrack_model = NanoTrack(
                backbone_param_path="./models/nanotrack_backbone_sim.param",
                backbone_bin_path="./models/nanotrack_backbone_sim.bin",
                head_param_path="./models/nanotrack_head_sim.param",
                head_bin_path="./models/nanotrack_head_sim.bin",
                fixed_roi_size=FIXED_ROI_SIZE  # 고정 ROI 크기 설정
            )
            print(f"NanoTrack models loaded successfully with fixed ROI size: {FIXED_ROI_SIZE}x{FIXED_ROI_SIZE}")
            USE_NANOTRACK = True
        except Exception as e:
            print(f"Warning: Could not load NanoTrack models: {e}")
            print("Falling back to template matching tracker")
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

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.1)
                continue
            
            original_frame = cv2.resize(frame, (1280, 720))
            
            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level += 0.5
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level -= 0.5
                zoom_command = None
            
            current_frame = original_frame.copy()
            
            process_new_coordinate(original_frame)
            
            if not target_selected:
                tracking = False
                center_x = 0
                center_y = 0

            if tracking and tracker is not None:
                try:
                    success, new_bbox = tracker.update(original_frame)
                    if success:
                        bbox = new_bbox
                        center_x = bbox[0] + bbox[2] // 2
                        center_y = bbox[1] + bbox[3] // 2
                    else:
                        tracking = False
                except Exception as e:
                    print(f"Tracker error: {e}")
                    tracking = False
            
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
            
            if tracking and bbox is not None:
                x, y, w, h = bbox
                
                disp_x, disp_y = original_to_display_coord(x, y)
                disp_x2, disp_y2 = original_to_display_coord(x + w, y + h)
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y
                
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                
                send_data_to_serial(center_x, center_y, tracking)
            
            # 상태 표시
            tracker_type = "NanoTrack" if USE_NANOTRACK else "Template"
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
```