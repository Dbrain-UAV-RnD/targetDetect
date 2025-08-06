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

# NCNN Tracker Class
class NCNNTracker:
    def __init__(self, backbone_param, backbone_bin, head_param, head_bin, use_gpu=False):
        """
        NCNN 기반 NanoTrack 트래커 초기화
        backbone_param: backbone .param 파일 경로
        backbone_bin: backbone .bin 파일 경로
        head_param: head .param 파일 경로
        head_bin: head .bin 파일 경로
        use_gpu: GPU 사용 여부
        """
        # Backbone 네트워크 (특징 추출)
        self.backbone = ncnn.Net()
        self.backbone.opt.use_vulkan_compute = use_gpu
        self.backbone.opt.num_threads = 4
        self.backbone.load_param(backbone_param)
        self.backbone.load_model(backbone_bin)
        
        # Head 네트워크 (분류 및 회귀)
        self.head = ncnn.Net()
        self.head.opt.use_vulkan_compute = use_gpu
        self.head.opt.num_threads = 4
        self.head.load_param(head_param)
        self.head.load_model(head_bin)
        
        self.template_feature = None
        self.search_size = 255
        self.template_size = 127
        self.stride = 16  # NanoTrack default stride
        self.initialized = False
        self.window = None
        
    def init(self, frame, bbox):
        """
        트래커 초기화
        frame: 첫 프레임
        bbox: (x, y, w, h) 형태의 바운딩 박스
        """
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        # Template 추출
        context_amount = 0.5
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.template_size / s_z
        
        template = self._crop_and_resize(frame, cx, cy, self.template_size, s_z)
        
        # Template feature 추출 (backbone 사용)
        template_blob = self._preprocess(template)
        ex = self.backbone.create_extractor()
        ex.input("input", template_blob)
        _, self.template_feature = ex.extract("output")  # backbone output
        
        # 윈도우 생성 (코사인 윈도우)
        self.window = self._create_window()
        
        self.target_pos = np.array([cx, cy])
        self.target_size = np.array([w, h])
        self.s_z = s_z
        self.initialized = True
        
        return True
    
    def update(self, frame):
        """
        트래킹 업데이트
        frame: 현재 프레임
        return: (success, bbox)
        """
        if not self.initialized:
            return False, None
        
        # Search region 크기 계산
        context_amount = 0.5
        wc_x = self.target_size[0] + context_amount * sum(self.target_size)
        hc_x = self.target_size[1] + context_amount * sum(self.target_size)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = self.search_size / s_x
        
        # Search region 추출
        search = self._crop_and_resize(frame, self.target_pos[0], self.target_pos[1],
                                      self.search_size, s_x)
        
        # Search feature 추출 (backbone 사용)
        search_blob = self._preprocess(search)
        ex_backbone = self.backbone.create_extractor()
        ex_backbone.input("input", search_blob)
        _, search_feature = ex_backbone.extract("output")
        
        # Head 네트워크로 예측
        ex_head = self.head.create_extractor()
        ex_head.input("template", self.template_feature)
        ex_head.input("search", search_feature)
        
        # Classification과 regression 결과 추출
        _, cls_score = ex_head.extract("cls")
        _, bbox_pred = ex_head.extract("reg")
        
        # Score map 처리
        score_map = cls_score.reshape(17, 17)  # NanoTrack은 17x17 출력
        
        # 윈도우 적용
        if self.window is not None:
            score_map = score_map * self.window
        
        # 최대 점수 위치 찾기
        best_idx = np.argmax(score_map)
        y_idx = best_idx // 17
        x_idx = best_idx % 17
        
        # 바운딩 박스 예측값 추출
        bbox_delta = bbox_pred.reshape(4, 17, 17)[:, y_idx, x_idx]
        
        # 위치 업데이트
        disp = (np.array([x_idx, y_idx]) - 8) * self.stride / scale_x
        self.target_pos += disp
        
        # 크기 업데이트
        lr = 0.4  # learning rate
        self.target_size = self.target_size * (1 - lr) + \
                          self.target_size * np.exp(bbox_delta[2:]) * lr
        
        # 바운딩 박스 생성
        x = self.target_pos[0] - self.target_size[0] / 2
        y = self.target_pos[1] - self.target_size[1] / 2
        
        # 이미지 경계 체크
        h, w = frame.shape[:2]
        x = max(0, min(x, w - self.target_size[0]))
        y = max(0, min(y, h - self.target_size[1]))
        
        bbox = (int(x), int(y), int(self.target_size[0]), int(self.target_size[1]))
        
        return True, bbox
    
    def _create_window(self):
        """
        코사인 윈도우 생성
        """
        hanning = np.hanning(17)
        window = np.outer(hanning, hanning)
        return window
    
    def _crop_and_resize(self, img, cx, cy, size, s):
        """
        이미지에서 특정 영역을 크롭하고 리사이즈
        """
        im_h, im_w = img.shape[:2]
        
        # 패딩 계산
        context_xmin = int(cx - s / 2)
        context_xmax = int(cx + s / 2)
        context_ymin = int(cy - s / 2)
        context_ymax = int(cy + s / 2)
        
        left_pad = max(0, -context_xmin)
        top_pad = max(0, -context_ymin)
        right_pad = max(0, context_xmax - im_w)
        bottom_pad = max(0, context_ymax - im_h)
        
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad
        
        # 패딩 적용
        if any([left_pad, top_pad, right_pad, bottom_pad]):
            img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))  # NanoTrack uses gray padding
        
        # 크롭 및 리사이즈
        patch = img[context_ymin:context_ymax, context_xmin:context_xmax]
        patch = cv2.resize(patch, (size, size))
        
        return patch
    
    def _preprocess(self, img):
        """
        이미지 전처리 (NanoTrack 스타일)
        """
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet 표준)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Create ncnn mat
        mat = ncnn.Mat(img.shape[2], img.shape[1], img.shape[0], img.data)
        
        return mat

# Global variables
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
    global latest_point, new_point_received, tracker, tracking, roi, current_frame, kalman, zoom_level
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
            
            # NCNN 트래커 초기화
            success = tracker.init(frame, roi)
            if success:
                tracking = True
                
                # Kalman filter 초기화
                kalman = cv2.KalmanFilter(4, 2)
                kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
                kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1

                center_x = left + (right - left) / 2
                center_y = top + (bottom - top) / 2
                kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
                kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        else:
            pass

def main():
    global current_frame, tracker, tracking, roi, target_selected, kalman, serial_port, zoom_level, zoom_command, zoom_center
    
    # NCNN NanoTrack 트래커 초기화
    # models 폴더에 있는 NanoTrack 모델 파일 사용
    MODEL_PATH = "./models"  # models 폴더 경로
    tracker = NCNNTracker(
        backbone_param=f"{MODEL_PATH}/nanotrack_backbone_sim.param",
        backbone_bin=f"{MODEL_PATH}/nanotrack_backbone_sim.bin",
        head_param=f"{MODEL_PATH}/nanotrack_head_sim.param",
        head_bin=f"{MODEL_PATH}/nanotrack_head_sim.bin",
        use_gpu=True  # Jetson의 GPU 사용
    )
    
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

            if tracking and tracker is not None and kalman is not None:
                prediction = kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                # NCNN 트래커 업데이트
                success, bbox = tracker.update(original_frame)
                if success:
                    x, y, w, h = bbox
                    roi = (x, y, w, h)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                    kalman.correct(measurement)
                else:
                    center_x, center_y = pred_x, pred_y
                    tracking = False  # 트래킹 실패 시 중지
            
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
                
                send_data_to_serial(center_x, center_y, tracking)
            
            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking else (0, 0, 255), 2)
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "NCNN Tracker", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if zoom_applied and tracking:
                disp_x, disp_y = original_to_display_coord(int(center_x), int(center_y))
                debug_info = f"Orig: ({int(center_x)}, {int(center_y)}) -> Disp: ({disp_x}, {disp_y})"
                cv2.putText(display_frame, debug_info, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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