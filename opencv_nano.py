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

# NCNN Python bindings - try different import methods
try:
    import ncnn
    print("Using ncnn module")
except ImportError:
    try:
        from ncnn import ncnn
        print("Using ncnn.ncnn module")
    except ImportError:
        print("Please install ncnn-python:")
        print("  pip install ncnn")
        print("  or")
        print("  pip install ncnn-vulkan")
        print("  or for ARM platforms:")
        print("  pip install ncnn-arm")
        exit(1)

# Set GStreamer plugin path
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

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

class NanoTrack:
    def __init__(self, model_path="models", use_gpu=False):
        self.cfg = {
            'context_amount': 0.5,
            'exemplar_size': 127,
            'instance_size': 255,
            'score_size': 16,
            'total_stride': 16,
            'window_influence': 0.21,
            'penalty_k': 0.04,
            'lr': 0.33
        }
        
        try:
            # Load models with basic settings
            self.net_backbone = ncnn.Net()
            
            # Set only the commonly available options
            self.net_backbone.opt.num_threads = 1  # Set threads before loading param
            
            # Try to enable GPU if available and requested
            if use_gpu:
                try:
                    if hasattr(self.net_backbone.opt, 'use_vulkan_compute'):
                        self.net_backbone.opt.use_vulkan_compute = True
                        print("Using GPU acceleration (Vulkan)")
                except:
                    print("GPU acceleration not available")
            
            self.net_backbone.load_param(f"{model_path}/nanotrack_backbone_sim.param")
            self.net_backbone.load_model(f"{model_path}/nanotrack_backbone_sim.bin")
            
            self.net_head = ncnn.Net()
            self.net_head.opt.num_threads = 1  # Set threads before loading param
            
            if use_gpu:
                try:
                    if hasattr(self.net_head.opt, 'use_vulkan_compute'):
                        self.net_head.opt.use_vulkan_compute = True
                except:
                    pass
                
            self.net_head.load_param(f"{model_path}/nanotrack_head_sim.param")
            self.net_head.load_model(f"{model_path}/nanotrack_head_sim.bin")
            
        except Exception as e:
            print(f"Failed to load NanoTrack models: {e}")
            raise
        
        self.state = {}
        self.window = None
        self.grid_to_search_x = None
        self.grid_to_search_y = None
        self.zf = None
        
        self._create_window()
        self._create_grids()
    
    def _create_window(self):
        """Create Hanning window for penalty"""
        score_size = self.cfg['score_size']
        hanning = np.hanning(score_size)
        self.window = np.outer(hanning, hanning).flatten()
    
    def _create_grids(self):
        """Create search grid coordinates"""
        sz = self.cfg['score_size']
        self.grid_to_search_x = np.zeros((sz, sz))
        self.grid_to_search_y = np.zeros((sz, sz))
        
        for i in range(sz):
            for j in range(sz):
                self.grid_to_search_x[i, j] = j * self.cfg['total_stride']
                self.grid_to_search_y[i, j] = i * self.cfg['total_stride']
        
        self.grid_to_search_x = self.grid_to_search_x.flatten()
        self.grid_to_search_y = self.grid_to_search_y.flatten()
    
    def _get_subwindow_tracking(self, im, pos, model_sz, original_sz, channel_ave):
        """Get subwindow for tracking"""
        c = (original_sz + 1) / 2
        context_xmin = int(np.round(pos[0] - c))
        context_xmax = context_xmin + original_sz - 1
        context_ymin = int(np.round(pos[1] - c))
        context_ymax = context_ymin + original_sz - 1
        
        left_pad = max(0, -context_xmin)
        top_pad = max(0, -context_ymin)
        right_pad = max(0, context_xmax - im.shape[1] + 1)
        bottom_pad = max(0, context_ymax - im.shape[0] + 1)
        
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad
        
        if any([top_pad, left_pad, right_pad, bottom_pad]):
            te_im = cv2.copyMakeBorder(im, top_pad, bottom_pad, left_pad, right_pad,
                                       cv2.BORDER_CONSTANT, value=channel_ave)
            im_path_original = te_im[context_ymin:context_ymax + 1, 
                                    context_xmin:context_xmax + 1]
        else:
            im_path_original = im[context_ymin:context_ymax + 1,
                                 context_xmin:context_xmax + 1]
        
        im_path = cv2.resize(im_path_original, (model_sz, model_sz))
        return im_path
    
    def init(self, img, bbox):
        """Initialize tracker with first frame and bounding box"""
        target_pos = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
        target_sz = np.array([bbox[2], bbox[3]])
        
        wc_z = target_sz[0] + self.cfg['context_amount'] * sum(target_sz)
        hc_z = target_sz[1] + self.cfg['context_amount'] * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        
        avg_chans = np.mean(img, axis=(0, 1))
        z_crop = self._get_subwindow_tracking(img, target_pos, 
                                              self.cfg['exemplar_size'],
                                              int(s_z), avg_chans)
        
        # Convert to NCNN format and extract features
        # Ensure z_crop is contiguous and uint8
        z_crop = np.ascontiguousarray(z_crop, dtype=np.uint8)
        
        # Create NCNN Mat - API may vary by version
        try:
            # Try newer API
            ncnn_img = ncnn.Mat.from_pixels(z_crop, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                                            z_crop.shape[1], z_crop.shape[0])
        except:
            try:
                # Try without PixelType enum
                ncnn_img = ncnn.Mat.from_pixels(z_crop, ncnn.Mat.PIXEL_BGR2RGB, 
                                                z_crop.shape[1], z_crop.shape[0])
            except:
                try:
                    # Try alternative API
                    ncnn_img = ncnn.Mat(z_crop.shape[1], z_crop.shape[0], z_crop.shape[2])
                    ncnn_img.from_pixels(z_crop, ncnn.Mat.PIXEL_BGR2RGB)
                except:
                    # Fallback to basic Mat creation
                    ncnn_img = ncnn.Mat(z_crop)
        
        ex_backbone = self.net_backbone.create_extractor()
        
        ex_backbone.input("input", ncnn_img)
        
        # Extract features - API may return different formats
        result = ex_backbone.extract("output")
        if isinstance(result, tuple):
            _, self.zf = result
        else:
            self.zf = result
        
        self.state = {
            'channel_ave': avg_chans,
            'im_h': img.shape[0],
            'im_w': img.shape[1],
            'target_pos': target_pos,
            'target_sz': target_sz
        }
    
    def _sigmoid(self, x):
        """Fast sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _sz_wh_fun(self, wh):
        """Size function"""
        pad = sum(wh) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)
    
    def _update(self, x_crops, target_pos, target_sz, scale_z):
        """Update tracker with new frame"""
        # Extract features from search region
        # Ensure x_crops is contiguous and uint8
        x_crops = np.ascontiguousarray(x_crops, dtype=np.uint8)
        
        # Create NCNN Mat - API may vary by version
        try:
            # Try newer API
            ncnn_img = ncnn.Mat.from_pixels(x_crops, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                            x_crops.shape[1], x_crops.shape[0])
        except:
            try:
                # Try without PixelType enum
                ncnn_img = ncnn.Mat.from_pixels(x_crops, ncnn.Mat.PIXEL_BGR2RGB,
                                                x_crops.shape[1], x_crops.shape[0])
            except:
                try:
                    # Try alternative API
                    ncnn_img = ncnn.Mat(x_crops.shape[1], x_crops.shape[0], x_crops.shape[2])
                    ncnn_img.from_pixels(x_crops, ncnn.Mat.PIXEL_BGR2RGB)
                except:
                    # Fallback to basic Mat creation
                    ncnn_img = ncnn.Mat(x_crops)
        
        ex_backbone = self.net_backbone.create_extractor()
        
        ex_backbone.input("input", ncnn_img)
        
        # Extract features - API may return different formats
        result = ex_backbone.extract("output")
        if isinstance(result, tuple):
            _, xf = result
        else:
            xf = result
        
        # Head network
        ex_head = self.net_head.create_extractor()
        
        ex_head.input("input1", self.zf)
        ex_head.input("input2", xf)
        
        # Extract classification and bbox prediction
        cls_result = ex_head.extract("output1")
        if isinstance(cls_result, tuple):
            _, cls_score = cls_result
        else:
            cls_score = cls_result
            
        bbox_result = ex_head.extract("output2")
        if isinstance(bbox_result, tuple):
            _, bbox_pred = bbox_result
        else:
            bbox_pred = bbox_result
        
        # Convert to numpy arrays
        cls_score_np = np.array(cls_score).reshape(self.cfg['score_size'], 
                                                   self.cfg['score_size'], 2)
        cls_score_sigmoid = self._sigmoid(cls_score_np[:, :, 1].flatten())
        
        bbox_pred_np = np.array(bbox_pred).reshape(self.cfg['score_size'],
                                                   self.cfg['score_size'], 4)
        
        # Calculate predicted bbox
        rows, cols = self.cfg['score_size'], self.cfg['score_size']
        pred_x1 = self.grid_to_search_x - bbox_pred_np[:, :, 0].flatten()
        pred_y1 = self.grid_to_search_y - bbox_pred_np[:, :, 1].flatten()
        pred_x2 = self.grid_to_search_x + bbox_pred_np[:, :, 2].flatten()
        pred_y2 = self.grid_to_search_y + bbox_pred_np[:, :, 3].flatten()
        
        # Size penalty
        w = pred_x2 - pred_x1
        h = pred_y2 - pred_y1
        
        sz_wh = self._sz_wh_fun(target_sz)
        s_c = np.maximum(w + h, 1e-6)
        r_c = np.maximum(w / np.maximum(h, 1e-6), 1e-6)
        
        penalty = np.exp(-(s_c * r_c - 1) * self.cfg['penalty_k'])
        
        # Window penalty
        pscore = penalty * cls_score_sigmoid * (1 - self.cfg['window_influence']) + \
                self.window * self.cfg['window_influence']
        
        # Get max score position
        max_idx = np.argmax(pscore)
        r_max = max_idx // cols
        c_max = max_idx % cols
        
        # Get real size
        pred_x1_real = pred_x1[max_idx]
        pred_y1_real = pred_y1[max_idx]
        pred_x2_real = pred_x2[max_idx]
        pred_y2_real = pred_y2[max_idx]
        
        pred_xs = (pred_x1_real + pred_x2_real) / 2
        pred_ys = (pred_y1_real + pred_y2_real) / 2
        pred_w = pred_x2_real - pred_x1_real
        pred_h = pred_y2_real - pred_y1_real
        
        diff_xs = pred_xs - self.cfg['instance_size'] / 2
        diff_ys = pred_ys - self.cfg['instance_size'] / 2
        
        diff_xs /= scale_z
        diff_ys /= scale_z
        pred_w /= scale_z
        pred_h /= scale_z
        
        target_sz = target_sz / scale_z
        
        # Learning rate
        lr = penalty[max_idx] * cls_score_sigmoid[max_idx] * self.cfg['lr']
        
        # Update position and size
        res_x = target_pos[0] + diff_xs
        res_y = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]
        
        target_pos[0] = res_x
        target_pos[1] = res_y
        target_sz[0] = target_sz[0] * (1 - lr) + lr * res_w
        target_sz[1] = target_sz[1] * (1 - lr) + lr * res_h
        
        return target_pos, target_sz, cls_score_sigmoid[max_idx]
    
    def track(self, img):
        """Track object in new frame"""
        target_pos = self.state['target_pos'].copy()
        target_sz = self.state['target_sz'].copy()
        
        hc_z = target_sz[1] + self.cfg['context_amount'] * sum(target_sz)
        wc_z = target_sz[0] + self.cfg['context_amount'] * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.cfg['exemplar_size'] / s_z
        
        d_search = (self.cfg['instance_size'] - self.cfg['exemplar_size']) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        
        x_crop = self._get_subwindow_tracking(img, target_pos,
                                              self.cfg['instance_size'],
                                              int(s_x), self.state['channel_ave'])
        
        # Update
        target_sz = target_sz * scale_z
        target_pos, target_sz, score = self._update(x_crop, target_pos, target_sz, scale_z)
        
        # Clip to image boundaries
        target_pos[0] = np.clip(target_pos[0], 0, self.state['im_w'])
        target_pos[1] = np.clip(target_pos[1], 0, self.state['im_h'])
        target_sz[0] = np.clip(target_sz[0], 10, self.state['im_w'])
        target_sz[1] = np.clip(target_sz[1], 10, self.state['im_h'])
        
        self.state['target_pos'] = target_pos
        self.state['target_sz'] = target_sz
        
        # Convert to bbox format (x, y, w, h)
        bbox = [target_pos[0] - target_sz[0]/2, 
                target_pos[1] - target_sz[1]/2,
                target_sz[0], target_sz[1]]
        
        return True, bbox, score

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
    global latest_point, new_point_received, tracker, tracking, roi, current_frame, kalman, zoom_level
    if new_point_received:
        new_point_received = False
        x, y = latest_point
        current_frame = frame.copy()
        
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            roi_size = int(500 / zoom_level)
            left = max(0, x - roi_size // 2)
            top = max(0, y - roi_size // 2)
            right = min(frame.shape[1], x + roi_size // 2)
            bottom = min(frame.shape[0], y + roi_size // 2)
            roi = (left, top, right - left, bottom - top)
            
            # Initialize NanoTrack instead of KCF
            tracker = NanoTrack(model_path="models", use_gpu=False)  # Set use_gpu=True for GPU acceleration
            tracker.init(frame, roi)
            tracking = True
            
            # Still use Kalman filter for prediction
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

                # Use NanoTrack for tracking
                success, bbox, score = tracker.track(original_frame)
                if success and score > 0.5:  # Add confidence threshold
                    x, y, w, h = [int(v) for v in bbox]
                    roi = (x, y, w, h)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                    kalman.correct(measurement)
                else:
                    center_x, center_y = pred_x, pred_y
                    tracking = False  # Lost tracking
            
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
            
            if zoom_applied and tracking:
                disp_x, disp_y = original_to_display_coord(int(center_x), int(center_y))
                debug_info = f"Orig: ({int(center_x)}, {int(center_y)}) -> Disp: ({disp_x}, {disp_y})"
                cv2.putText(display_frame, debug_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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