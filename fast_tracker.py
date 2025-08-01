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

class FastCorrelationTracker:
    """고속 상관 필터 기반 트래커 - DCF (Discriminative Correlation Filter) 변형"""
    def __init__(self):
        self.learning_rate = 0.02
        self.sigma = 0.5
        self.cell_size = 4
        self.padding = 1.5
        self.output_sigma_factor = 0.1
        self.scale_step = 1.05
        self.num_scales = 3
        
        self.model_alphaf = None
        self.model_xf = None
        self.roi = None
        
    def init(self, frame, bbox):
        """트래커 초기화"""
        x, y, w, h = bbox
        self.roi = bbox
        self.size = (w, h)
        self.center = (x + w/2, y + h/2)
        
        # 패치 추출
        patch = self._get_subwindow(frame, self.center, self.size)
        
        # HOG 특징 추출
        features = self._extract_hog_features(patch)
        
        # 가우시안 레이블 생성
        label = self._gaussian_label(features.shape[:2])
        label_f = np.fft.fft2(label)
        
        # 모델 초기화
        self.model_xf = np.fft.fft2(features, axes=(0,1))
        kf = self._gaussian_correlation(self.model_xf, self.model_xf)
        self.model_alphaf = label_f / (kf + 1e-4)
        
        return True
        
    def update(self, frame):
        """새 프레임에서 객체 추적"""
        # 멀티스케일 검색
        best_score = -np.inf
        best_scale = 1.0
        best_pos = self.center
        
        scales = self.scale_step ** np.arange(-self.num_scales//2 + 1, self.num_scales//2 + 1)
        
        for scale in scales:
            # 현재 스케일에서 패치 추출
            scaled_size = (self.size[0] * scale, self.size[1] * scale)
            patch = self._get_subwindow(frame, self.center, scaled_size)
            
            # 원래 크기로 리사이즈
            patch = cv2.resize(patch, (int(self.size[0]), int(self.size[1])))
            
            # HOG 특징 추출
            features = self._extract_hog_features(patch)
            zf = np.fft.fft2(features, axes=(0,1))
            
            # 응답 계산
            kf = self._gaussian_correlation(zf, self.model_xf)
            response = np.real(np.fft.ifft2(self.model_alphaf * kf))
            
            # 최대값 찾기
            max_val = np.max(response)
            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                max_pos = np.unravel_index(np.argmax(response), response.shape)
                
                # 서브픽셀 정확도
                if max_pos[0] > 0 and max_pos[0] < response.shape[0]-1:
                    best_pos = (
                        self.center[0] + (max_pos[1] - response.shape[1]//2) * self.cell_size * scale,
                        self.center[1] + (max_pos[0] - response.shape[0]//2) * self.cell_size * scale
                    )
        
        # 위치 업데이트
        self.center = best_pos
        self.size = (self.size[0] * best_scale, self.size[1] * best_scale)
        
        # 모델 업데이트
        patch = self._get_subwindow(frame, self.center, self.size)
        features = self._extract_hog_features(patch)
        xf_new = np.fft.fft2(features, axes=(0,1))
        kf_new = self._gaussian_correlation(xf_new, xf_new)
        alphaf_new = self._create_label(features.shape[:2]) / (kf_new + 1e-4)
        
        # 점진적 업데이트
        self.model_xf = (1 - self.learning_rate) * self.model_xf + self.learning_rate * xf_new
        self.model_alphaf = (1 - self.learning_rate) * self.model_alphaf + self.learning_rate * alphaf_new
        
        # 바운딩 박스 반환
        bbox = (
            int(self.center[0] - self.size[0]/2),
            int(self.center[1] - self.size[1]/2),
            int(self.size[0]),
            int(self.size[1])
        )
        
        return True, bbox
        
    def _get_subwindow(self, frame, center, size):
        """중심점과 크기로 패치 추출"""
        w, h = int(size[0] * self.padding), int(size[1] * self.padding)
        x, y = int(center[0] - w/2), int(center[1] - h/2)
        
        # 경계 처리
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        patch = frame[y1:y2, x1:x2]
        
        # 패딩 처리
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            patch_new = np.zeros((h, w, 3), dtype=frame.dtype)
            px1 = max(0, -x)
            py1 = max(0, -y)
            px2 = px1 + (x2 - x1)
            py2 = py1 + (y2 - y1)
            patch_new[py1:py2, px1:px2] = patch
            patch = patch_new
            
        return patch
        
    def _extract_hog_features(self, patch):
        """간단한 HOG 특징 추출"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # 리사이즈
        resized = cv2.resize(gray, (32, 32))
        
        # 간단한 그래디언트 계산
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
        
        # 매그니튜드와 방향
        mag = np.sqrt(gx**2 + gy**2)
        
        # 정규화
        features = mag / (np.max(mag) + 1e-6)
        
        return features
        
    def _gaussian_label(self, shape):
        """가우시안 레이블 생성"""
        output_sigma = np.sqrt(np.prod(shape)) * self.output_sigma_factor
        grid_y, grid_x = np.ogrid[:shape[0], :shape[1]]
        center_y, center_x = shape[0]//2, shape[1]//2
        d = ((grid_x - center_x)**2 + (grid_y - center_y)**2)
        label = np.exp(-d / (2 * output_sigma**2))
        return label
        
    def _create_label(self, shape):
        """FFT 도메인에서 레이블 생성"""
        label = self._gaussian_label(shape)
        return np.fft.fft2(label)
        
    def _gaussian_correlation(self, xf, yf):
        """가우시안 커널 상관관계"""
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(xf * np.conj(xf)).sum() / N
        yy = np.real(yf * np.conj(yf)).sum() / N
        xyf = xf * np.conj(yf)
        xy = np.real(np.fft.ifft2(xyf))
        
        kf = np.fft.fft2(np.exp(-1 / self.sigma**2 * np.maximum(0, (xx + yy - 2 * xy))))
        return kf

# Global variables
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
    global latest_point, new_point_received, tracker, tracking, roi, current_frame, zoom_level, center_x, center_y
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
            
            tracker = FastCorrelationTracker()
            tracker.init(frame, roi)
            tracking = True

def main():
    global current_frame, tracker, tracking, roi, target_selected, serial_port, zoom_level, zoom_command, zoom_center
    
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
            frame = original_frame.copy()
            
            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level += 0.5
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level -= 0.5
                zoom_command = None
            
            current_frame = original_frame.copy()
            
            process_new_coordinate(original_frame)
            
            center_x, center_y = 0, 0
            
            if not target_selected:
                tracking = False

            if tracking and tracker is not None:
                success, bbox = tracker.update(original_frame)
                if success:
                    x, y, w, h = bbox
                    roi = (x, y, w, h)
                    center_x = x + w / 2
                    center_y = y + h / 2
            
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
            
            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking else (0, 0, 255), 2)
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Fast DCF Tracker", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
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
        print(f"Error: {e}")
    
    finally:
        if serial_port and serial_port.is_open:
            serial_port.close()
        pipeline.set_state(Gst.State.NULL)
        cap.release()

if __name__ == "__main__":
    main()