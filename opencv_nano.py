import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
import os
import ncnn

# Set GStreamer plugin path
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

current_frame = None
roi = None
tracker = None
tracking = False
kalman = None
latest_point = None
new_point_received = False
target_selected = False
serial_port = None
zoom_level = 1.0
zoom_command = None
zoom_center = None
failed_frames = 0
max_failed_frames = 10  # 추가: 연속 실패 프레임 제한

class NanoTracker:
    def __init__(self, backbone_param, backbone_bin, head_param, head_bin):
        # NCNN 모델 로드
        self.backbone = ncnn.Net()
        self.backbone.load_param(backbone_param)
        self.backbone.load_model(backbone_bin)
        
        self.head = ncnn.Net()
        self.head.load_param(head_param)
        self.head.load_model(head_bin)
        
        self.template_feat = None
        self.window = None
        
    def init(self, frame, roi):
        """KCF와 동일한 인터페이스"""
        x, y, w, h = roi
        
        # 템플릿 추출 (127x127)
        template = self._crop_and_resize(frame, x, y, w, h, 127)
        
        # Backbone으로 특징 추출
        ex = self.backbone.create_extractor()
        mat_in = ncnn.Mat.from_pixels(template, ncnn.Mat.RGB, 127, 127)
        ex.input("data", mat_in)
        _, self.template_feat = ex.extract("output")
        
        # 윈도우 생성 (코사인 윈도우)
        self.window = self._create_window(255)
        self.center = [x + w/2, y + h/2]
        self.size = [w, h]
        
        return True
    
    def update(self, frame):
        """추적 업데이트"""
        # 검색 영역 추출 (255x255)
        search = self._crop_and_resize(
            frame, 
            self.center[0] - self.size[0]/2,
            self.center[1] - self.size[1]/2,
            self.size[0], self.size[1], 
            255
        )
        
        # Backbone으로 검색 영역 특징 추출
        ex = self.backbone.create_extractor()
        mat_in = ncnn.Mat.from_pixels(search, ncnn.Mat.RGB, 255, 255)
        ex.input("data", mat_in)
        _, search_feat = ex.extract("output")
        
        # Head로 상관관계 계산
        ex_head = self.head.create_extractor()
        ex_head.input("template", self.template_feat)
        ex_head.input("search", search_feat)
        _, score_map = ex_head.extract("output")
        
        # Score map을 numpy로 변환
        score = np.array(score_map).reshape(255, 255)
        
        # 윈도우 적용 및 최대값 찾기
        score = score * self.window
        max_idx = np.unravel_index(score.argmax(), score.shape)
        
        # 좌표 업데이트
        displacement = [(max_idx[1] - 127) * self.size[0] / 255,
                       (max_idx[0] - 127) * self.size[1] / 255]
        
        self.center[0] += displacement[0]
        self.center[1] += displacement[1]
        
        # bbox 반환
        bbox = (
            self.center[0] - self.size[0]/2,
            self.center[1] - self.size[1]/2,
            self.size[0],
            self.size[1]
        )
        
        return True, bbox
    
    def _crop_and_resize(self, img, x, y, w, h, out_size):
        """ROI 추출 및 리사이즈"""
        crop = img[int(y):int(y+h), int(x):int(x+w)]
        return cv2.resize(crop, (out_size, out_size))
    
    def _create_window(self, size):
        """코사인 윈도우 생성"""
        hanning = np.hanning(size)
        window = np.outer(hanning, hanning)
        return window

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

    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

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
    
    # 이전 값을 저장하기 위한 변수들 추가
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
            
            # 값이 이전과 같은지 확인
            if (x == prev_x and y == prev_y and 
                is_target_selected == prev_target_selected and 
                zoom_cmd == prev_zoom_cmd):
                continue  # 이전과 동일한 값이면 처리 건너뛰기
            
            # 현재 값을 이전 값으로 저장
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
    global latest_point, new_point_received, tracker, tracking, roi, current_frame, kalman, zoom_level, failed_frames
    if new_point_received:
        new_point_received = False
        x, y = latest_point
        current_frame = frame.copy()
        
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            roi_size = int(100 / zoom_level)  # zoom_level 고려 ROI 크기
            half_size = roi_size // 2
            left = max(0, x - half_size)
            top = max(0, y - half_size)
            right = min(frame.shape[1], x + half_size)
            bottom = min(frame.shape[0], y + half_size)
            
            # 경계에서 ROI 크기 유지 (고정 크기 강제)
            if right - left < roi_size:
                if left == 0:
                    right = min(roi_size, frame.shape[1])
                else:
                    left = max(0, right - roi_size)
            
            if bottom - top < roi_size:
                if top == 0:
                    bottom = min(roi_size, frame.shape[0])
                else:
                    top = max(0, bottom - roi_size)
            
            roi = (left, top, right - left, bottom - top)
            
            # tracker = cv2.TrackerKCF_create()

            tracker = NanoTracker(
                "./models/nanotrack_backbone_sim.param",
                "./models/nanotrack_backbone_sim.bin",
                "./models/nanotrack_head_sim.param",
                "./models/nanotrack_head_sim.bin"
            )

            tracker.init(frame, roi)
            tracking = True
            failed_frames = 0  # 실패 카운터 리셋
            
            kalman = cv2.KalmanFilter(4, 2)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.01  # process noise 줄임 (더 부드럽게)
            kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.5  # measurement noise 증가 (측정값 덜 신뢰)

            center_x = left + (right - left) / 2
            center_y = top + (bottom - top) / 2
            kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
            kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        else:
            pass

def main():
    global current_frame, tracker, tracking, roi, target_selected, kalman, serial_port, zoom_level, zoom_command, zoom_center, failed_frames
    
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
    
    last_serial_time = 0
    serial_interval = 1.0 / 25.0
    
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
            pred_x, pred_y = 0, 0
            
            if not target_selected:
                tracking = False
                failed_frames = 0

            if tracking and tracker is not None and kalman is not None:
                prediction = kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                success, bbox = tracker.update(original_frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # bbox 클립 (이미지 밖으로 나가지 않게)
                    x = max(0, min(x, original_frame.shape[1] - w))
                    y = max(0, min(y, original_frame.shape[0] - h))
                    roi = (x, y, w, h)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                    kalman.correct(measurement)
                    failed_frames = 0  # 성공 시 리셋
                else:
                    failed_frames += 1
                    center_x, center_y = pred_x, pred_y
                    # 예측 위치도 클립
                    center_x = max(0, min(center_x, original_frame.shape[1]))
                    center_y = max(0, min(center_y, original_frame.shape[0]))
                
                # 연속 실패 시 트래킹 중지
                if failed_frames > max_failed_frames:
                    tracking = False
                    failed_frames = 0
            
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
                
                disp_pred_x, disp_pred_y = original_to_display_coord(pred_x, pred_y)
                cv2.circle(display_frame, (disp_pred_x, disp_pred_y), 5, (255, 0, 0), -1)
                
                # serial limit
                # current_time = time.time()
                # if current_time - last_serial_time >= serial_interval:
                #     send_data_to_serial(center_x, center_y, tracking)
                #     last_serial_time = current_time
                
                send_data_to_serial(center_x, center_y, tracking)
            
            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking else (0, 0, 255), 2)
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # if zoom_center is not None:
            #     if 'original_to_display_coord' in locals():
            #         disp_zoom_x, disp_zoom_y = original_to_display_coord(*zoom_center)
            #         cv2.circle(display_frame, (disp_zoom_x, disp_zoom_y), 8, (255, 255, 0), -1)
            #         cv2.putText(display_frame, f"Zoom Center: {zoom_center} -> ({disp_zoom_x}, {disp_zoom_y})", 
            #                     (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if zoom_applied and tracking:
                disp_x, disp_y = original_to_display_coord(int(center_x), int(center_y))
                
                # test_x, test_y = local_display_to_original_coord(disp_x, disp_y)
                
                debug_info = f"Orig: ({int(center_x)}, {int(center_y)}) -> Disp: ({disp_x}, {disp_y})"
                cv2.putText(display_frame, debug_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # conversion_test = f"Back conversion test: ({disp_x}, {disp_y}) -> ({test_x}, {test_y})"
                # cv2.putText(display_frame, conversion_test, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
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
        pass
    
    finally:
        if serial_port and serial_port.is_open:
            serial_port.close()
        pipeline.set_state(Gst.State.NULL)
        cap.release()

if __name__ == "__main__":
    main()