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
import ncnn

os.environ['GST_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

class NanoTrack:
    
    def __init__(self, backbone_param_path="./models/nanotrack_backbone_sim.param",
                 backbone_bin_path="./models/nanotrack_backbone_sim.bin",
                 head_param_path="./models/nanotrack_head_sim.param",
                 head_bin_path="./models/nanotrack_head_sim.bin"):
        
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
        
        self.net_backbone = ncnn.Net()
        self.net_backbone.opt.num_threads = 2
        try:
            self.net_backbone.opt.use_vulkan_compute = False
        except:
            pass
        self.net_backbone.load_param(backbone_param_path)
        self.net_backbone.load_model(backbone_bin_path)
        
        self.net_head = ncnn.Net()
        self.net_head.opt.num_threads = 2
        try:
            self.net_head.opt.use_vulkan_compute = False
        except:
            pass
        self.net_head.load_param(head_param_path)
        self.net_head.load_model(head_bin_path)
        
        self.center = None
        self.target_sz = None
        self.bbox = None
        self.zf = None
        self.channel_ave = None
        self.im_h = 0
        self.im_w = 0
        
        self.failed_frames = 0
        self.max_failed_frames = 10
        self.confidence_threshold = 0.2
        
        self._create_window()
        self._create_grids()
        
        self.ex_backbone = self.net_backbone.create_extractor()
        self.ex_head = self.net_head.create_extractor()
        
    def _create_window(self):
        score_size = self.cfg['score_size']
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        
    def _create_grids(self):
        sz = self.cfg['score_size']
        x, y = np.meshgrid(np.arange(sz), np.arange(sz))
        self.grid_to_search_x = x.flatten() * self.cfg['total_stride']
        self.grid_to_search_y = y.flatten() * self.cfg['total_stride']
        
    def _get_subwindow_tracking(self, im, pos, model_sz, original_sz, avg_chans):
        c = (original_sz + 1) // 2
        context_xmin = int(pos[0] - c)
        context_xmax = context_xmin + original_sz - 1
        context_ymin = int(pos[1] - c)
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
            
        if im_patch.shape[:2] != (model_sz, model_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz), interpolation=cv2.INTER_LINEAR)
        
        return im_patch
    
    def init(self, frame, point):
        x, y = int(point[0]), int(point[1])
        roi_size = 100
        half_size = roi_size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        w = x2 - x1
        h = y2 - y1
        
        if w < 60 or h < 60:
            roi_size = 60
            half_size = roi_size // 2
            x1 = max(0, min(x - half_size, frame.shape[1] - roi_size))
            y1 = max(0, min(y - half_size, frame.shape[0] - roi_size))
            w = h = roi_size
        
        self.bbox = (x1, y1, w, h)
        self.center = [x1 + w//2, y1 + h//2]
        self.target_sz = [w, h]
        
        self.im_h = frame.shape[0]
        self.im_w = frame.shape[1]
        self.channel_ave = cv2.mean(frame)[:3]
        
        wc_z = self.target_sz[0] + self.cfg['context_amount'] * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.cfg['context_amount'] * sum(self.target_sz)
        s_z = int(math.sqrt(wc_z * hc_z))
        
        z_crop = self._get_subwindow_tracking(frame, self.center, 
                                             self.cfg['exemplar_size'], 
                                             s_z, self.channel_ave)
        
        z_crop_rgb = cv2.cvtColor(z_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(z_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 
                                      z_crop.shape[1], z_crop.shape[0])
        
        self.ex_backbone.input("input", mat_in)
        _, self.zf = self.ex_backbone.extract("output")
        
        self.failed_frames = 0
        
        return True
        
    def update(self, frame):
        if self.center is None:
            return False, self.bbox
            
        wc_z = self.target_sz[0] + self.cfg['context_amount'] * sum(self.target_sz)
        hc_z = self.target_sz[1] + self.cfg['context_amount'] * sum(self.target_sz)
        s_z = math.sqrt(wc_z * hc_z)
        scale_z = self.cfg['exemplar_size'] / s_z
        
        d_search = (self.cfg['instance_size'] - self.cfg['exemplar_size']) / 2
        pad = d_search / scale_z
        s_x = int(s_z + 2 * pad)
        
        x_crop = self._get_subwindow_tracking(frame, self.center,
                                             self.cfg['instance_size'],
                                             s_x, self.channel_ave)
        
        x_crop_rgb = cv2.cvtColor(x_crop, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(x_crop_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                      x_crop.shape[1], x_crop.shape[0])
        
        self.ex_backbone.input("input", mat_in)
        _, xf = self.ex_backbone.extract("output")
        
        self.ex_head.input("input1", self.zf)
        self.ex_head.input("input2", xf)
        
        _, cls_score = self.ex_head.extract("output1")
        _, bbox_pred = self.ex_head.extract("output2")
        
        cls_score_np = np.array(cls_score)[1, :, :].flatten()
        cls_score_sigmoid = 1 / (1 + np.exp(-cls_score_np))
        
        bbox_pred_np = np.array(bbox_pred).reshape(4, -1)
        
        pred_x1 = self.grid_to_search_x - bbox_pred_np[0]
        pred_y1 = self.grid_to_search_y - bbox_pred_np[1]
        pred_x2 = self.grid_to_search_x + bbox_pred_np[2]
        pred_y2 = self.grid_to_search_y + bbox_pred_np[3]
        
        w = pred_x2 - pred_x1
        h = pred_y2 - pred_y1
        
        target_sz_prod = math.sqrt((self.target_sz[0] + sum(self.target_sz) * 0.5) * 
                                  (self.target_sz[1] + sum(self.target_sz) * 0.5))
        
        s_c = np.maximum(w / target_sz_prod, target_sz_prod / w) * \
              np.maximum(h / target_sz_prod, target_sz_prod / h)
        
        ratio = self.target_sz[0] / (self.target_sz[1] + 1e-6)
        r_c = np.maximum(ratio / (w / (h + 1e-6)), (w / (h + 1e-6)) / ratio)
        
        penalty = np.exp(-(s_c * r_c - 1) * self.cfg['penalty_k'])
        pscore = penalty * cls_score_sigmoid * (1 - self.cfg['window_influence']) + \
                self.window * self.cfg['window_influence']
        
        best_idx = np.argmax(pscore)
        best_score = cls_score_sigmoid[best_idx]
        
        if best_score < self.confidence_threshold:
            self.failed_frames += 1
            if self.failed_frames > self.max_failed_frames:
                return False, self.bbox
            return True, self.bbox
        
        self.failed_frames = 0
        
        pred_xs = (pred_x1[best_idx] + pred_x2[best_idx]) / 2
        pred_ys = (pred_y1[best_idx] + pred_y2[best_idx]) / 2
        pred_w = pred_x2[best_idx] - pred_x1[best_idx]
        pred_h = pred_y2[best_idx] - pred_y1[best_idx]
        
        diff_xs = (pred_xs - self.cfg['instance_size'] / 2) / scale_z
        diff_ys = (pred_ys - self.cfg['instance_size'] / 2) / scale_z
        
        lr = penalty[best_idx] * best_score * self.cfg['lr']
        
        self.center[0] = np.clip(self.center[0] + diff_xs, 0, self.im_w)
        self.center[1] = np.clip(self.center[1] + diff_ys, 0, self.im_h)
        
        self.target_sz[0] = np.clip(self.target_sz[0] * (1 - lr) + pred_w / scale_z * lr, 10, self.im_w)
        self.target_sz[1] = np.clip(self.target_sz[1] * (1 - lr) + pred_h / scale_z * lr, 10, self.im_h)
        
        self.bbox = (
            int(self.center[0] - self.target_sz[0] / 2),
            int(self.center[1] - self.target_sz[1] / 2),
            int(self.target_sz[0]),
            int(self.target_sz[1])
        )
        
        return True, self.bbox


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
nanotrack_model = None

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SERIAL_SCALE_X = 2.0 / FRAME_WIDTH
SERIAL_SCALE_Y = 2.0 / FRAME_HEIGHT

def camera_init(capture, resolution_index=0):
    if not capture.isOpened():
        return False
    
    resolutions = [(1920, 1080), (3840, 2160), (4208, 3120)]
    if resolution_index < 0 or resolution_index >= len(resolutions):
        return False
        
    frame_width, frame_height = resolutions[resolution_index]
    
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FPS, 25)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return True

def setup_serial():
    try:
        ser = serial.Serial(
            port='/dev/serial0',
            baudrate=57600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        return ser
    except Exception as e:
        print(f"Serial port error: {e}")
        return None

def send_data_to_serial(x, y, is_tracking):
    global serial_port
    
    if serial_port is None or not serial_port.is_open:
        return
    
    x_norm = int((-1 + x * SERIAL_SCALE_X) * 1000)
    y_norm = int((1 - y * SERIAL_SCALE_Y) * 1000)
    
    try:
        data = struct.pack('<BBhhB', 
                          0xBB, 0x88,
                          x_norm, y_norm,
                          0xFF if is_tracking else 0x00)
        
        checksum = 0
        for byte in data:
            checksum ^= byte
        
        serial_port.write(data + bytes([checksum]))
        
    except Exception as e:
        print(f"Serial send error: {e}")
        try:
            if serial_port:
                serial_port.close()
            time.sleep(0.5)
            serial_port = setup_serial()
        except:
            pass

def udp_receiver():
    global latest_point, new_point_received, target_selected, zoom_command, tracking, zoom_center
    
    udp_socket = None
    for _ in range(30):
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.bind(('192.168.10.219', 5001))
            break
        except:
            time.sleep(5)
    
    if not udp_socket:
        print("Failed to bind UDP socket")
        return
    
    prev_data = None
    
    try:
        while True:
            data, _ = udp_socket.recvfrom(8)
            
            if len(data) < 8 or data[0] != 0xAA or data[1] != 0x77:
                continue
            
            if data == prev_data:
                continue
            prev_data = data
            
            x = struct.unpack('<H', data[2:4])[0]
            y = struct.unpack('<H', data[4:6])[0]
            is_target_selected = data[6] == 0xFF
            zoom_cmd = data[7]
            
            if x >= FRAME_WIDTH or y >= FRAME_HEIGHT:
                continue
            
            if is_target_selected and not target_selected:
                latest_point = (x, y)
                new_point_received = True
                target_selected = True
            elif not is_target_selected and target_selected:
                target_selected = False
                tracking = False
            
            if zoom_cmd == 0x02:
                zoom_command = 'zoom_in'
                zoom_center = (x, y)
            elif zoom_cmd == 0x01:
                zoom_command = 'zoom_out'
                zoom_center = (x, y)
            else:
                zoom_command = None
                
    except Exception as e:
        print(f"UDP receiver error: {e}")
    finally:
        if udp_socket:
            udp_socket.close()

def process_frame_tracking(frame):
    global latest_point, new_point_received, tracker, tracking
    global center_x, center_y, bbox, nanotrack_model
    
    if new_point_received:
        new_point_received = False
        point = latest_point
        
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            if nanotrack_model.init(frame, point):
                tracking = True
                bbox = nanotrack_model.bbox
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
            else:
                tracking = False
    
    if not target_selected:
        tracking = False
        center_x = center_y = 0
        return
    
    if tracking and nanotrack_model:
        try:
            success, new_bbox = nanotrack_model.update(frame)
            if success:
                bbox = new_bbox
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
            else:
                tracking = False
        except Exception as e:
            print(f"Tracking error: {e}")
            tracking = False

def apply_zoom(frame, zoom_level, zoom_center):
    if zoom_level <= 1.0 or zoom_center is None:
        return frame, None
    
    h, w = frame.shape[:2]
    
    zoom_width = int(w / zoom_level)
    zoom_height = int(h / zoom_level)
    
    x1 = np.clip(zoom_center[0] - zoom_width // 2, 0, w - zoom_width)
    y1 = np.clip(zoom_center[1] - zoom_height // 2, 0, h - zoom_height)
    
    roi = frame[y1:y1+zoom_height, x1:x1+zoom_width]
    zoomed = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed, (x1, y1, zoom_level)

def draw_overlay(frame, bbox, tracking, target_selected, zoom_level):
    if tracking and bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_active = (0, 255, 0)
    color_inactive = (0, 0, 255)
    
    texts = []
    if tracking:
        texts.append((f"NanoTrack: X={center_x}, Y={center_y}", color_active))
    else:
        texts.append(("NanoTrack: OFF", color_inactive))
    
    texts.append((f"Target: {'Selected' if target_selected else 'None'}", 
                 color_active if target_selected else color_inactive))
    texts.append((f"Zoom: {zoom_level:.1f}x", color_active))
    
    y_pos = 30
    for text, color in texts:
        cv2.putText(frame, text, (10, y_pos), font, 0.7, color, 2)
        y_pos += 30

def main():
    global tracker, tracking, target_selected, serial_port
    global zoom_level, zoom_command, zoom_center, center_x, center_y, bbox
    global nanotrack_model
    
    try:
        nanotrack_model = NanoTrack(
            backbone_param_path="./models/nanotrack_backbone_sim.param",
            backbone_bin_path="./models/nanotrack_backbone_sim.bin",
            head_param_path="./models/nanotrack_head_sim.param",
            head_bin_path="./models/nanotrack_head_sim.bin"
        )
    except Exception as e:
        print(f"Failed to load NanoTrack models: {e}")
        return
    
    serial_port = setup_serial()
    
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()
    
    cap = cv2.VideoCapture(0)
    if not camera_init(cap):
        print("Camera initialization failed")
        return
    
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
    pipeline.set_state(Gst.State.PLAYING)
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    zoom_info = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level = min(zoom_level + 0.5, 3.0)
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level = max(zoom_level - 0.5, 1.0)
                zoom_command = None
            
            process_frame_tracking(frame)
            
            if tracking:
                send_data_to_serial(center_x, center_y, True)
            
            if zoom_level > 1.0 and zoom_center:
                display_frame, zoom_info = apply_zoom(frame, zoom_level, zoom_center)
            else:
                display_frame = frame
                zoom_info = None
            
            if zoom_info and tracking and bbox:
                x1, y1, scale = zoom_info
                x, y, w, h = bbox
                
                disp_x = int((x - x1) * scale)
                disp_y = int((y - y1) * scale)
                disp_w = int(w * scale)
                disp_h = int(h * scale)
                
                cv2.rectangle(display_frame, (disp_x, disp_y), 
                            (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
            elif tracking and bbox:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            draw_overlay(display_frame, None, tracking, target_selected, zoom_level)
            
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_time)
                fps_time = current_time
                
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            buffer = Gst.Buffer.new_allocate(None, display_frame.nbytes, None)
            buffer.fill(0, display_frame.tobytes())
            buffer.pts = Gst.CLOCK_TIME_NONE
            buffer.dts = Gst.CLOCK_TIME_NONE
            appsrc.emit("push-buffer", buffer)
            
            if os.path.exists("stop.signal"):
                os.remove("stop.signal")
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if serial_port and serial_port.is_open:
            serial_port.close()
        pipeline.set_state(Gst.State.NULL)
        cap.release()

if __name__ == "__main__":
    main()