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

class RobustPitchTracker:
    """Robust tracker for rapid pitch down movements"""
    
    def __init__(self):
        # Basic settings
        self.template = None
        self.template_gray = None
        self.bbox = None
        self.center = None
        
        # Tracking parameters
        self.search_scale = 5.0  # Expand search area to 3x template size
        self.confidence_threshold = 0.3  # Low threshold setting
        self.template_update_rate = 0.05
        
        # Rapid movement handling
        self.max_movement_ratio = 0.5  # Allow movement up to 50% of template size per frame
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_decay = 0.8  # Velocity decay
        
        # Failure handling
        self.failed_frames = 0
        self.max_failed_frames = 10  # More lenient setting
        
        # Pitch down detection
        self.prev_center_y = 0
        self.pitch_down_threshold = 50  # Consider pitch down if moving 50+ pixels down per frame
        
    def simple_roi_selection(self, frame, point, roi_size=100):
        """Simple and reliable ROI selection"""
        x, y = int(point[0]), int(point[1])
        
        # Square ROI centered on click point
        half_size = roi_size // 2
        
        # Boundary check
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        # Calculate actual size
        actual_w = x2 - x1
        actual_h = y2 - y1
        
        # Ensure minimum size
        if actual_w < 60 or actual_h < 60:
            roi_size = 60
            half_size = roi_size // 2
            x1 = max(0, min(x - half_size, frame.shape[1] - roi_size))
            y1 = max(0, min(y - half_size, frame.shape[0] - roi_size))
            actual_w = actual_h = roi_size
        
        print(f"ROI selected: ({x1}, {y1}, {actual_w}, {actual_h}) at click point ({x}, {y})")
        return (x1, y1, actual_w, actual_h)
    
    def init(self, frame, point):
        """Initialize tracker"""
        # Simple ROI selection
        self.bbox = self.simple_roi_selection(frame, point)
        x, y, w, h = self.bbox
        
        # Extract template
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Set center point
        self.center = [x + w//2, y + h//2]
        self.prev_center_y = self.center[1]
        
        # Initialize velocity
        self.velocity_x = 0
        self.velocity_y = 0
        self.failed_frames = 0
        
        print(f"Tracker initialized at {self.center} with bbox {self.bbox}")
        return True
    
    def update(self, frame):
        """Update tracker - considering rapid pitch down"""
        if self.template is None or self.template_gray is None:
            return False, self.bbox
        
        # Velocity-based predicted position
        predicted_x = self.center[0] + self.velocity_x
        predicted_y = self.center[1] + self.velocity_y
        
        # Set search area - very wide
        template_w, template_h = self.template.shape[1], self.template.shape[0]
        search_w = int(template_w * self.search_scale)
        search_h = int(template_h * self.search_scale)
        
        # Expand search area downward considering rapid downward movement
        search_x1 = max(0, int(predicted_x - search_w // 2))
        search_y1 = max(0, int(predicted_y - search_h // 4))  # Only 1/4 upward
        search_x2 = min(frame.shape[1], search_x1 + search_w)
        search_y2 = min(frame.shape[0], search_y1 + int(search_h * 1.5))  # 1.5x downward
        
        search_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            self.failed_frames += 1
            return self.failed_frames <= self.max_failed_frames, self.bbox
        
        # Convert to grayscale
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Template matching with multiple scales
        best_score = -1
        best_loc = None
        best_scale = 1.0
        
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # Wider scale range
        
        for scale in scales:
            # Resize template
            scaled_w = int(self.template_gray.shape[1] * scale)
            scaled_h = int(self.template_gray.shape[0] * scale)
            
            if scaled_w > search_gray.shape[1] or scaled_h > search_gray.shape[0]:
                continue
                
            scaled_template = cv2.resize(self.template_gray, (scaled_w, scaled_h))
            
            # Template matching
            result = cv2.matchTemplate(search_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_scale = scale
        
        # Confidence check
        if best_score < self.confidence_threshold:
            self.failed_frames += 1
            
            # Velocity-based position prediction (considering rapid pitch down)
            if self.failed_frames <= self.max_failed_frames:
                # Apply pitch down acceleration
                if self.velocity_y > 0:  # If moving downward
                    self.velocity_y *= 1.2  # Accelerate
                
                self.center[0] += self.velocity_x
                self.center[1] += self.velocity_y
                
                # Boundary check
                self.center[0] = max(template_w//2, min(self.center[0], frame.shape[1] - template_w//2))
                self.center[1] = max(template_h//2, min(self.center[1], frame.shape[0] - template_h//2))
                
                # Update bounding box
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
        
        # Successful matching
        self.failed_frames = 0
        
        # Calculate new position
        template_w_scaled = int(self.template_gray.shape[1] * best_scale)
        template_h_scaled = int(self.template_gray.shape[0] * best_scale)
        
        new_center_x = search_x1 + best_loc[0] + template_w_scaled // 2
        new_center_y = search_y1 + best_loc[1] + template_h_scaled // 2
        
        # Calculate velocity
        self.velocity_x = new_center_x - self.center[0]
        self.velocity_y = new_center_y - self.center[1]
        
        # Detect pitch down
        y_movement = new_center_y - self.prev_center_y
        if y_movement > self.pitch_down_threshold:
            print(f"Pitch down detected: {y_movement} pixels")
            # More aggressive tracking during pitch down
            self.confidence_threshold = 0.2
            self.search_scale = 4.0
        else:
            # Return to normal state
            self.confidence_threshold = 0.3
            self.search_scale = 3.0
        
        self.prev_center_y = new_center_y
        
        # Update center point
        self.center[0] = new_center_x
        self.center[1] = new_center_y
        
        # Velocity decay
        self.velocity_x *= self.velocity_decay
        self.velocity_y *= self.velocity_decay
        
        # Update bounding box
        new_w = int(template_w_scaled)
        new_h = int(template_h_scaled)
        
        self.bbox = (
            int(self.center[0] - new_w//2),
            int(self.center[1] - new_h//2),
            new_w,
            new_h
        )
        
        # Template update (only when high confidence)
        if best_score > 0.6:
            x, y, w, h = self.bbox
            if 0 <= x < frame.shape[1] - w and 0 <= y < frame.shape[0] - h:
                new_template = frame[y:y+h, x:x+w]
                if new_template.size > 0:
                    self.template = cv2.addWeighted(
                        self.template, 1 - self.template_update_rate,
                        cv2.resize(new_template, (self.template.shape[1], self.template.shape[0])),
                        self.template_update_rate, 0
                    )
                    self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        return True, self.bbox

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
    global center_x, center_y, bbox
    
    if new_point_received:
        new_point_received = False
        point = latest_point
        current_frame = frame.copy()
        
        print(f"Processing new coordinate: {point}")
        
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            tracker = RobustPitchTracker()
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
        else:
            print(f"Point {point} is out of bounds")

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
            
            if tracking and bbox is not None:
                x, y, w, h = bbox
                
                disp_x, disp_y = original_to_display_coord(x, y)
                disp_x2, disp_y2 = original_to_display_coord(x + w, y + h)
                disp_w = disp_x2 - disp_x
                disp_h = disp_y2 - disp_y
                
                # Bounding box
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                
                send_data_to_serial(center_x, center_y, tracking)
            
            # Status information
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