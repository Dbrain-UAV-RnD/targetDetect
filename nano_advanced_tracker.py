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

class SmartNanoTracker:
    """지능형 NANO 트래커 - 객체 크기 자동 감지 및 적응형 추적"""
    
    def __init__(self):
        self.template = None
        self.template_gray = None
        self.bbox = None
        self.center = None
        self.scale_factor = 1.0
        self.confidence_threshold = 0.6
        self.search_region_scale = 2.5  # 템플릿 크기의 2.5배 영역에서 검색
        self.template_update_rate = 0.05  # 템플릿 업데이트 비율
        self.scale_tolerance = 0.2  # 스케일 변화 허용 범위
        
        # 멀티스케일 검색
        self.scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # 성능 최적화
        self.skip_frame_count = 0
        self.max_skip_frames = 2  # 최대 2프레임 스킵
        
    def detect_object_around_point(self, frame, point, search_radius=100):
        """지정된 점 주변에서 객체의 실제 경계 감지"""
        x, y = point
        
        # 검색 영역 설정
        search_x1 = max(0, x - search_radius)
        search_y1 = max(0, y - search_radius)
        search_x2 = min(frame.shape[1], x + search_radius)
        search_y2 = min(frame.shape[0], y + search_radius)
        
        search_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            # 기본 ROI 반환
            default_size = 80
            return (x - default_size//2, y - default_size//2, default_size, default_size)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # 적응적 임계값으로 이진화
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 에지 기반 검출 시도
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 클릭 지점과 가장 가까운 컨투어 찾기
            click_point_local = (x - search_x1, y - search_y1)
            best_contour = None
            min_distance = float('inf')
            
            for contour in contours:
                # 컨투어 면적이 너무 작거나 크면 제외
                area = cv2.contourArea(contour)
                if area < 100 or area > (search_radius * 2) ** 2 * 0.8:
                    continue
                
                # 컨투어 중심점과의 거리 계산
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    dist = np.sqrt((cx - click_point_local[0])**2 + (cy - click_point_local[1])**2)
                    
                    if dist < min_distance:
                        min_distance = dist
                        best_contour = contour
            
            if best_contour is not None:
                # 바운딩 박스 계산
                bx, by, bw, bh = cv2.boundingRect(best_contour)
                
                # 글로벌 좌표로 변환
                global_x = search_x1 + bx
                global_y = search_y1 + by
                
                # 최소/최대 크기 제한
                bw = max(30, min(bw, 300))
                bh = max(30, min(bh, 300))
                
                return (global_x, global_y, bw, bh)
        
        # 객체를 찾지 못한 경우 기본 ROI
        default_size = min(120, search_radius)
        return (x - default_size//2, y - default_size//2, default_size, default_size)
    
    def init(self, frame, point):
        """트래커 초기화 - 클릭 지점에서 객체 자동 감지"""
        # 객체 경계 자동 감지
        self.bbox = self.detect_object_around_point(frame, point)
        x, y, w, h = self.bbox
        
        # 템플릿 추출
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # 중심점 계산
        self.center = (x + w//2, y + h//2)
        
        print(f"Initialized tracker with bbox: {self.bbox}")
        return True
    
    def update(self, frame):
        """프레임에서 객체 추적"""
        if self.template is None:
            return False, self.bbox
        
        # 프레임 스킵 최적화
        self.skip_frame_count += 1
        if self.skip_frame_count < self.max_skip_frames:
            return True, self.bbox
        self.skip_frame_count = 0
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 검색 영역 설정
        search_w = int(self.bbox[2] * self.search_region_scale)
        search_h = int(self.bbox[3] * self.search_region_scale)
        
        search_x1 = max(0, self.center[0] - search_w // 2)
        search_y1 = max(0, self.center[1] - search_h // 2)
        search_x2 = min(frame.shape[1], search_x1 + search_w)
        search_y2 = min(frame.shape[0], search_y1 + search_h)
        
        search_region = frame_gray[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            return False, self.bbox
        
        best_match = None
        best_confidence = 0
        best_scale = 1.0
        best_location = None
        
        # 멀티스케일 템플릿 매칭
        for scale in self.scales:
            # 템플릿 스케일링
            scaled_template = cv2.resize(self.template_gray, 
                                       (int(self.template_gray.shape[1] * scale),
                                        int(self.template_gray.shape[0] * scale)))
            
            if (scaled_template.shape[0] > search_region.shape[0] or 
                scaled_template.shape[1] > search_region.shape[1]):
                continue
            
            # 템플릿 매칭
            result = cv2.matchTemplate(search_region, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_scale = scale
                best_location = max_loc
                best_match = scaled_template
        
        # 신뢰도 체크
        if best_confidence < self.confidence_threshold:
            print(f"Low confidence: {best_confidence:.3f}")
            return False, self.bbox
        
        # 새로운 위치 계산
        if best_location is not None:
            new_w = int(self.bbox[2] * best_scale)
            new_h = int(self.bbox[3] * best_scale)
            
            new_x = search_x1 + best_location[0]
            new_y = search_y1 + best_location[1]
            
            # 경계 체크
            new_x = max(0, min(new_x, frame.shape[1] - new_w))
            new_y = max(0, min(new_y, frame.shape[0] - new_h))
            
            self.bbox = (new_x, new_y, new_w, new_h)
            self.center = (new_x + new_w//2, new_y + new_h//2)
            
            # 템플릿 점진적 업데이트
            if best_confidence > 0.8:  # 높은 신뢰도일 때만 업데이트
                new_template = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                if new_template.size > 0:
                    # 가중 평균으로 템플릿 업데이트
                    self.template = cv2.addWeighted(self.template, 1 - self.template_update_rate,
                                                  cv2.resize(new_template, (self.template.shape[1], self.template.shape[0])), 
                                                  self.template_update_rate, 0)
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
            print(f"Initializing tracker at point: {point}")
            
            tracker = SmartNanoTracker()
            success = tracker.init(frame, point)
            
            if success:
                tracking = True
                bbox = tracker.bbox
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                print(f"Tracker initialized successfully with bbox: {bbox}")
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
            
            # 새 좌표 처리
            process_new_coordinate(original_frame)
            
            if not target_selected:
                tracking = False
                center_x = 0
                center_y = 0

            # 트래킹 업데이트
            if tracking and tracker is not None:
                try:
                    success, new_bbox = tracker.update(original_frame)
                    if success:
                        bbox = new_bbox
                        center_x = bbox[0] + bbox[2] // 2
                        center_y = bbox[1] + bbox[3] // 2
                    else:
                        print("Tracking failed")
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
            
            # 트래킹 결과 표시
            if tracking and bbox is not None:
                x, y, w, h = bbox
                
                disp_x, disp_y = original_to_display_coord(x, y)
                disp_x2, disp_y2 = original_to_display_coord(x + w, y + h)
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y
                
                # 바운딩 박스 그리기
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                
                # 중심점 그리기
                disp_center_x, disp_center_y = original_to_display_coord(int(center_x), int(center_y))
                cv2.circle(display_frame, (disp_center_x, disp_center_y), 5, (0, 0, 255), -1)
                
                # 크로스헤어 그리기
                cv2.line(display_frame, (disp_center_x-10, disp_center_y), (disp_center_x+10, disp_center_y), (255, 0, 0), 2)
                cv2.line(display_frame, (disp_center_x, disp_center_y-10), (disp_center_x, disp_center_y+10), (255, 0, 0), 2)
                
                send_data_to_serial(center_x, center_y, tracking)
            
            # 상태 정보 표시
            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking else (0, 0, 255), 2)
            
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_selected else (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Zoom: {zoom_level}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "NANO Smart Tracker", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if bbox is not None and tracking:
                bbox_info = f"Size: {bbox[2]}x{bbox[3]}"
                cv2.putText(display_frame, bbox_info, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
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