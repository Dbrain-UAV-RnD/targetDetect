import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
import os
import torch
from ultralytics import SAM, FastSAM

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
max_failed_frames = 10
model = None
segmented_frame = None
prev_frame = None
tracking_points = None
frame_count = 0
resegment_interval = 5

lk_params = dict(winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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

    # fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = capture.get(cv2.CAP_PROP_FPS)

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

def resegment_from_tracking_points(frame, points):
    global model
    if len(points) < 3:
        return None, None
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    if len(points.shape) == 3:
        center_x = int(np.mean(points[:, 0, 0]))
        center_y = int(np.mean(points[:, 0, 1]))
    else:
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
    
    results = model.predict(frame, points=[[center_x, center_y]], labels=[1], device=device, verbose=False)
    if len(results[0].masks.data) > 0:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        
        feature_params = dict(maxCorners=6,
                             qualityLevel=0.01,
                             minDistance=7,
                             blockSize=3,
                             mask=mask)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        
        return new_corners, results[0].plot()
    return None, None

def process_new_coordinate(frame):
    global latest_point, new_point_received, tracker, tracking, roi, current_frame, kalman, zoom_level, failed_frames
    global model, segmented_frame, prev_frame, tracking_points, frame_count
    
    if new_point_received:
        new_point_received = False
        x, y = latest_point
        current_frame = frame.copy()
        
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            device = 0 if torch.cuda.is_available() else "cpu"
            results = model.predict(current_frame, points=[[x, y]], labels=[1], device=device, verbose=False)
            segmented_frame = results[0].plot()
            
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            
            feature_params = dict(maxCorners=6,
                                 qualityLevel=0.01,
                                 minDistance=3,
                                 blockSize=7,
                                 mask=mask)
            
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
            
            if corners is not None:
                tracking_points = corners
                prev_frame = gray_frame.copy()
                tracking = True
                frame_count = 0
                failed_frames = 0

def main():
    global current_frame, tracker, tracking, roi, target_selected, kalman, serial_port, zoom_level, zoom_command, zoom_center, failed_frames
    global model, segmented_frame, prev_frame, tracking_points, frame_count
    
    serial_port = setup_serial()
    
    # Initialize SAM model
    # model = SAM("mobile_sam.pt")
    model = FastSAM("FastSAM-s.pt")
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
        "udpsink host=192.168.10.201 port=10010 buffer-size=2097152 sync=true async=false"
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

            if tracking and tracking_points is not None and prev_frame is not None:
                gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                
                next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, tracking_points, None, **lk_params)
                
                good_new = next_pts[status == 1]
                good_old = tracking_points[status == 1]
                
                if len(good_new) > 2:
                    frame_count += 1
                    
                    if frame_count % resegment_interval == 0:
                        new_corners, new_segmented = resegment_from_tracking_points(original_frame, good_new)
                        if new_corners is not None:
                            tracking_points = new_corners
                            segmented_frame = new_segmented
                        else:
                            tracking_points = good_new.reshape(-1, 1, 2)
                    else:
                        tracking_points = good_new.reshape(-1, 1, 2)
                    
                    # Calculate center from tracking points
                    if len(tracking_points.shape) == 3:
                        center_x = int(np.mean(tracking_points[:, 0, 0]))
                        center_y = int(np.mean(tracking_points[:, 0, 1]))
                    else:
                        center_x = int(np.mean(tracking_points[:, 0]))
                        center_y = int(np.mean(tracking_points[:, 1]))
                    
                    prev_frame = gray_frame.copy()
                    failed_frames = 0
                else:
                    tracking = False
                    tracking_points = None
                    failed_frames += 1
                
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
            
            if tracking and tracking_points is not None:
                # Draw tracking points
                for point in tracking_points:
                    if len(point.shape) == 2:
                        x, y = point.ravel().astype(int)
                    else:
                        x, y = point[0], point[1]
                    disp_x, disp_y = original_to_display_coord(x, y)
                    cv2.circle(display_frame, (disp_x, disp_y), 3, (0, 255, 0), -1)
                
                # Draw center point
                disp_center_x, disp_center_y = original_to_display_coord(int(center_x), int(center_y))
                cv2.circle(display_frame, (disp_center_x, disp_center_y), 5, (0, 0, 255), -1)
                
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