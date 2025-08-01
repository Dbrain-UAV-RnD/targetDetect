import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
import os

# Set GStreamer plugin path
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

class FinalTracker:
    """최종 트래커 - 간단한 초기화 + 강력한 추적 + 대형 피치 다운 대응"""
    
    def __init__(self):
        # Core tracking parameters (이전 optimized 방식 유지)
        self.exemplar_size = 127
        self.instance_size = 255
        self.context_amount = 0.5
        
        # Template and search region
        self.template = None
        self.template_gray = None
        self.bbox = None
        self.center = None
        self.target_sz = None
        
        # Tracking optimization parameters
        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_lr = 0.59
        self.response_up = 16
        
        # Multi-scale search
        self.scales = np.array([0.96, 0.98, 1.0, 1.02, 1.04])
        
        # Performance optimization
        self.skip_update_threshold = 0.7
        self.learning_rate = 0.012
        
        # Failure detection - 더 관대하게
        self.confidence_threshold = 0.25  # 더 낮은 임계값
        self.max_failed_frames = 15  # 더 많은 실패 허용
        self.failed_frames = 0
        
        # 대형 피치 다운 대응
        self.velocity_x = 0
        self.velocity_y = 0
        self.max_pitch_down = 200  # 한 프레임에 200픽셀까지 피치 다운 허용
        self.pitch_acceleration = 1.5  # 피치 다운 시 가속도
        
    def create_cosine_window(self, size):
        """코사인 윈도우 생성"""
        cos_window = np.hanning(size)
        return np.outer(cos_window, cos_window)
    
    def get_subwindow_tracking(self, im, pos, model_sz, original_sz):
        """서브윈도우 추출 - 이전 방식 유지"""
        if isinstance(pos, float):
            pos = [pos, pos]
        
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        
        context_xmin = round(pos[0] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)
        context_ymax = context_ymin + sz - 1
        
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
        
        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (im_sz[0] + top_pad + bottom_pad, im_sz[1] + left_pad + right_pad, im_sz[2])
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + im_sz[0], left_pad:left_pad + im_sz[1], :] = im
        else:
            te_im = im
        
        subwindow = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        
        if model_sz != original_sz:
            subwindow = cv2.resize(subwindow, (model_sz, model_sz))
        
        return subwindow
    
    def simple_init(self, frame, point, roi_size=100):
        """매우 간단한 초기화 - 클릭 지점 중심 정사각형"""
        x, y = int(point[0]), int(point[1])
        
        # 클릭 지점을 중심으로 고정 크기 정사각형
        half_size = roi_size // 2
        
        # 경계 체크
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # 실제 크기
        w = x2 - x1
        h = y2 - y1
        
        # 최소 크기 보장
        if w < 80 or h < 80:
            roi_size = 80
            half_size = roi_size // 2
            x1 = max(0, min(x - half_size, frame.shape[1] - roi_size))
            y1 = max(0, min(y - half_size, frame.shape[0] - roi_size))
            w = h = roi_size
        
        return (x1, y1, w, h)
    
    def init(self, frame, point):
        """트래커 초기화"""
        # 간단한 ROI 선택
        self.bbox = self.simple_init(frame, point)
        x, y, w, h = self.bbox
        
        # 중심점과 타겟 크기 설정
        self.center = np.array([x + w/2, y + h/2])
        self.target_sz = np.array([w, h])
        
        # 컨텍스트 크기 계산
        wc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        
        # 템플릿 추출
        z_crop = self.get_subwindow_tracking(frame, self.center, self.exemplar_size, s_z)
        self.template = z_crop
        self.template_gray = cv2.cvtColor(z_crop, cv2.COLOR_BGR2GRAY)
        
        # 속도 초기화
        self.velocity_x = 0
        self.velocity_y = 0
        self.failed_frames = 0
        
        print(f"Tracker initialized: bbox={self.bbox}, center={self.center}")
        return True
    
    def update(self, frame):
        """트래커 업데이트 - 대형 피치 다운 고려"""
        if self.template is None:
            return False, self.bbox
        
        # 컨텍스트 크기 계산
        wc_z = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        
        # 대형 피치 다운을 고려한 검색 영역 확장
        s_x = s_z * (self.instance_size / self.exemplar_size)
        
        # 피치 다운 예상 위치 계산
        predicted_center = self.center.copy()
        predicted_center[0] += self.velocity_x
        predicted_center[1] += self.velocity_y
        
        # 대형 피치 다운 시 검색 영역을 아래쪽으로 크게 확장
        if self.velocity_y > 50:  # 하향 속도가 클 때
            s_x_expanded = s_x * 2.0  # 검색 영역 2배 확장
            # 검색 중심을 아래쪽으로 이동
            predicted_center[1] += self.velocity_y * self.pitch_acceleration
        else:
            s_x_expanded = s_x * 1.5
        
        # 검색 영역 추출
        x_crop = self.get_subwindow_tracking(frame, predicted_center, 
                                           int(self.instance_size * 1.5), 
                                           round(s_x_expanded))
        
        if x_crop.size == 0:
            self.failed_frames += 1
            if self.failed_frames > self.max_failed_frames:
                return False, self.bbox
            
            # 속도 기반 예측으로 위치 업데이트
            self.center[0] += self.velocity_x
            self.center[1] += self.velocity_y
            
            # 대형 피치 다운 시 가속
            if self.velocity_y > 0:
                self.velocity_y *= self.pitch_acceleration
                
            # 바운딩 박스 업데이트
            self.bbox = (
                int(self.center[0] - self.target_sz[0]/2),
                int(self.center[1] - self.target_sz[1]/2),
                int(self.target_sz[0]),
                int(self.target_sz[1])
            )
            
            return True, self.bbox
        
        # 멀티스케일 템플릿 매칭
        best_score = -1
        best_loc = None
        best_scale_id = 0
        
        search_gray = cv2.cvtColor(x_crop, cv2.COLOR_BGR2GRAY)
        
        for i, scale in enumerate(self.scales):
            scaled_template = cv2.resize(self.template_gray, 
                                       (int(self.template_gray.shape[1] * scale),
                                        int(self.template_gray.shape[0] * scale)))
            
            if (scaled_template.shape[0] > search_gray.shape[0] or 
                scaled_template.shape[1] > search_gray.shape[1]):
                continue
            
            res = cv2.matchTemplate(search_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # 스케일 페널티 적용
            scale_penalty = np.exp(-np.abs(np.log(scale)) * self.scale_penalty)
            score = max_val * scale_penalty
            
            if score > best_score:
                best_score = score
                best_loc = max_loc
                best_scale_id = i
        
        # 신뢰도 체크
        if best_score < self.confidence_threshold:
            self.failed_frames += 1
            if self.failed_frames > self.max_failed_frames:
                return False, self.bbox
            
            # 실패 시에도 속도 기반으로 계속 추적
            self.center[0] += self.velocity_x
            self.center[1] += self.velocity_y
            
            # 대형 피치 다운 가속
            if self.velocity_y > 0:
                self.velocity_y = min(self.velocity_y * self.pitch_acceleration, self.max_pitch_down)
            
            self.bbox = (
                int(self.center[0] - self.target_sz[0]/2),
                int(self.center[1] - self.target_sz[1]/2),
                int(self.target_sz[0]),
                int(self.target_sz[1])
            )
            
            return True, self.bbox
        
        # 실패 카운터 리셋
        self.failed_frames = 0
        
        if best_loc is not None:
            best_scale = self.scales[best_scale_id]
            
            template_center_x = int(self.template_gray.shape[1] * best_scale / 2)
            template_center_y = int(self.template_gray.shape[0] * best_scale / 2)
            
            new_center_x = best_loc[0] + template_center_x
            new_center_y = best_loc[1] + template_center_y
            
            # 글로벌 좌표로 변환
            expanded_instance_size = int(self.instance_size * 1.5)
            scale_factor = s_x_expanded / expanded_instance_size
            dx = (new_center_x - expanded_instance_size/2) * scale_factor
            dy = (new_center_y - expanded_instance_size/2) * scale_factor
            
            # 속도 계산
            old_center = self.center.copy()
            new_center_global = predicted_center + np.array([dx, dy])
            
            self.velocity_x = new_center_global[0] - old_center[0]
            self.velocity_y = new_center_global[1] - old_center[1]
            
            # 대형 피치 다운 제한
            if self.velocity_y > self.max_pitch_down:
                self.velocity_y = self.max_pitch_down
            
            # 중심점 업데이트
            self.center = new_center_global
            
            # 타겟 크기 업데이트
            new_width = self.target_sz[0] * best_scale
            new_height = self.target_sz[1] * best_scale
            
            self.target_sz[0] = (1 - self.scale_lr) * self.target_sz[0] + self.scale_lr * new_width
            self.target_sz[1] = (1 - self.scale_lr) * self.target_sz[1] + self.scale_lr * new_height
            
            # 바운딩 박스 업데이트
            self.bbox = (
                int(self.center[0] - self.target_sz[0]/2),
                int(self.center[1] - self.target_sz[1]/2),
                int(self.target_sz[0]),
                int(self.target_sz[1])
            )
            
            # 템플릿 업데이트 (높은 신뢰도일 때만)
            if best_score > self.skip_update_threshold:
                new_s_z = round(np.sqrt((self.target_sz[0] + self.context_amount * sum(self.target_sz)) * 
                                      (self.target_sz[1] + self.context_amount * sum(self.target_sz))))
                new_template = self.get_subwindow_tracking(frame, self.center, self.exemplar_size, new_s_z)
                if new_template.size > 0:
                    self.template = cv2.addWeighted(
                        self.template, 1 - self.learning_rate,
                        cv2.resize(new_template, (self.template.shape[1], self.template.shape[0])),
                        self.learning_rate, 0
                    )
                    self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            
            return True, self.bbox
        
        return False, self.bbox

# Global variables
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
    global latest_point, new_point_received, tracker, tracking, current_frame
    global center_x, center_y, bbox
    
    if new_point_received:
        new_point_received = False
        point = latest_point
        current_frame = frame.copy()
        
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            tracker = FinalTracker()
            success = tracker.init(frame, point)
            
            if success:
                tracking = True
                bbox = tracker.bbox
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                print(f"Tracker initialized successfully")
            else:
                tracking = False
                print("Failed to initialize tracker")

def main():
    global current_frame, tracker, tracking, target_selected, serial_port
    global zoom_level, zoom_command, zoom_center, center_x, center_y, bbox
    
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
            
            # Zoom handling
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
            
            # 트래킹 결과 표시 - 간단하게
            if tracking and bbox is not None:
                x, y, w, h = bbox
                
                disp_x, disp_y = original_to_display_coord(x, y)
                disp_x2, disp_y2 = original_to_display_coord(x + w, y + h)
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y
                
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                
                send_data_to_serial(center_x, center_y, tracking)
            
            # 최소한의 상태 정보
            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
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