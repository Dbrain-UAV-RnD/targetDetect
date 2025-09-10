# sam_m_pi5_optimized.py
#
# Raspberry Pi 5–tuned version:
# - Keep OpenCV for core vision (Shi–Tomasi + LK optical flow)
# - Switch camera I/O to GStreamer + libcamerasrc (Bookworm camera stack)
# - Headless-friendly (no HighGUI used)
# - Unify FPS to 25 from end to end
# - Remove heavy debug drawing by default (toggle DEBUG_DRAW)
# - Eliminate 1280x720 hard-coding in serial normalization (uses frame shape)
# - Debounce UDP events and make socket non-blocking
# - Minimize color conversions and YOLO calls; use retina_masks=False for speed
# - Rely on appsrc do-timestamp for PTS/DTS (no manual stamping)
#
# Install:
#   pip install --upgrade opencv-python-headless ultralytics pyserial
#   sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libcamera \
#       gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

import os
os.environ['GST_PLUGIN_PATH'] = os.environ.get('GST_PLUGIN_PATH', '/usr/lib/aarch64-linux-gnu/gstreamer-1.0')
"zzzzzzzz"
import cv2
import numpy as np
import socket
import threading
import struct
import time
import serial
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import torch
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 25
RESEGMENT_INTERVAL = 8      # N frames between re-segmentation while tracking
DEBUG_DRAW   = False        # Heavy overlays (YOLO mask plots) off by default
YOLO_IMGSZ   = 640          # Power-of-32 input (speeds up CPU inference)

# LK / Shi–Tomasi parameters (leaner than defaults to reduce CPU)
MAX_CORNERS      = 4
FEATURE_PARAMS = dict(
    maxCorners=MAX_CORNERS,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=5,
    mask=None
)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

# UDP
UDP_HOST = '192.168.10.219'
UDP_PORT = 5001

# RTSP/UDP OUT
SINK_HOST = '192.168.10.201'
SINK_PORT = 10010
BITRATE_KBPS = 6000

# Serial: try common Raspberry Pi ports; fall back to USB
SERIAL_CANDIDATES = [
    '/dev/serial0',      # symlink to primary UART on Pi
    '/dev/ttyAMA0',
    '/dev/ttyS0',
    '/dev/ttyUSB0',
    '/dev/ttyUSB1'
]
SERIAL_BAUD = 57600

# Globals (kept small & intentional)
latest_point = None
new_point_received = False
target_selected = False
zoom_level = 1.0
zoom_command = None
zoom_center = None
tracking = False
prev_frame = None
tracking_points = None
frame_count = 0
failed_frames = 0
max_failed_frames = 10

serial_port = None
model = None

# Will be replaced at runtime with closure aware of zoom window
def display_to_original_coord(x, y):
    return int(x), int(y)


# -----------------------------
# Utilities
# -----------------------------
def setup_serial():
    for dev in SERIAL_CANDIDATES:
        try:
            ser = serial.Serial(
                port=dev,
                baudrate=SERIAL_BAUD,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            return ser
        except Exception:
            continue
    return None


def normalize_for_serial(px, py, width, height):
    # Map image pixel (0..W-1, 0..H-1) to [-1000, +1000] range as in original code
    xn = int((-1.0 + (px * (2.0 / max(1, width)))) * 1000.0)
    yn = int(( 1.0 - (py * (2.0 / max(1, height)))) * 1000.0)
    return xn, yn


def send_data_to_serial(px, py, is_tracking, frame_w, frame_h):
    global serial_port
    if serial_port is None or not serial_port.is_open:
        return

    x_norm, y_norm = normalize_for_serial(px, py, frame_w, frame_h)

    try:
        data = struct.pack('<BBhhB', 0xBB, 0x88, x_norm, y_norm, 0xFF if is_tracking else 0x00)
        checksum = 0
        for b in data:
            checksum ^= b
        data += bytes([checksum])
        serial_port.write(data)
    except Exception:
        try:
            if serial_port and serial_port.is_open:
                serial_port.close()
            time.sleep(0.25)
            # attempt reopen
            serial_port = setup_serial()
        except Exception:
            pass


def gstreamer_camera_pipeline(width, height, fps):
    # Try multiple camera pipeline options for better compatibility
    pipelines = [
        # Option 1: Try v4l2src with USB camera
        f"v4l2src device=/dev/video0 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false",
        # Option 2: Try v4l2src with different video device  
        f"v4l2src device=/dev/video1 ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false",
        # Option 3: Fallback libcamerasrc (if available)
        f"libcamerasrc ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false"
    ]
    return pipelines


def build_output_pipeline(width, height, fps, host, port, bitrate_kbps):
    # Software x264 on Pi5 (no HW encoder). Keep zerolatency + superfast + key-int=1
    return (
        "appsrc name=source is-live=true format=3 do-timestamp=true ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw ! "
        f"x264enc bitrate={bitrate_kbps} tune=zerolatency speed-preset=superfast key-int-max=1 ! "
        "h264parse ! "
        "rtph264pay config-interval=1 ! "
        "queue max-size-buffers=400 max-size-time=0 max-size-bytes=0 ! "
        f"udpsink host={host} port={port} buffer-size=2097152 sync=true async=false"
    )


# -----------------------------
# UDP receiver (non-blocking, de-duplicated)
# -----------------------------
def udp_receiver():
    global latest_point, new_point_received, target_selected, zoom_command, tracking, zoom_center

    prev_x = prev_y = None
    prev_target_selected = None
    prev_zoom_cmd = None

    # bind with retries
    max_retries, retry_delay = 30, 1.0
    for attempt in range(max_retries):
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.settimeout(0.5)
            udp_socket.bind((UDP_HOST, UDP_PORT))
            break
        except Exception:
            time.sleep(retry_delay)
    else:
        return

    try:
        while True:
            try:
                data, _ = udp_socket.recvfrom(8)
            except socket.timeout:
                continue
            except Exception:
                break

            if len(data) < 8 or data[0] != 0xAA or data[1] != 0x77:
                continue

            x = struct.unpack('<H', data[2:4])[0]
            y = struct.unpack('<H', data[4:6])[0]
            is_target = (data[6] == 0xFF)
            zoom_cmd = data[7]

            # dedupe
            if (x == prev_x and y == prev_y and is_target == prev_target_selected and zoom_cmd == prev_zoom_cmd):
                continue

            prev_x, prev_y = x, y
            prev_target_selected = is_target
            prev_zoom_cmd = zoom_cmd

            orig_x, orig_y = x, y
            try:
                orig_x, orig_y = display_to_original_coord(x, y)
            except Exception:
                pass

            if is_target:
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

    finally:
        try:
            udp_socket.close()
        except Exception:
            pass


# -----------------------------
# YOLO Segmentation helpers
# -----------------------------
def get_best_mask_from_point(results, x, y):
    try:
        masks = getattr(results[0], "masks", None)
        if masks is None or len(masks.data) == 0:
            return None

        # Prefer the mask that contains the point; else the largest by area
        best = None
        best_area = -1
        for mask in masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            h, w = mask_np.shape[:2]
            if 0 <= int(y) < h and 0 <= int(x) < w and mask_np[int(y), int(x)] > 0:
                return (mask_np * 255)
            area = mask_np.sum()
            if area > best_area:
                best_area = area
                best = (mask_np * 255)
        return best
    except Exception:
        return None


def resegment_from_tracking_points(frame_bgr, points, device):
    global model
    try:
        if points is None or len(points) < 3:
            return None, None

        if len(points.shape) == 3:
            cx = int(np.mean(points[:, 0, 0]))
            cy = int(np.mean(points[:, 0, 1]))
        else:
            cx = int(np.mean(points[:, 0]))
            cy = int(np.mean(points[:, 1]))

        results = model.predict(
            frame_bgr,
            device=device,
            verbose=False,
            retina_masks=False,
            imgsz=YOLO_IMGSZ
        )

        mask = get_best_mask_from_point(results, cx, cy)
        if mask is None:
            return None, None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if mask.shape[:2] != gray.shape[:2]:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))

        params = FEATURE_PARAMS.copy()
        params['mask'] = mask
        new_corners = cv2.goodFeaturesToTrack(gray, **params)

        overlay = results[0].plot() if DEBUG_DRAW else None
        return new_corners, overlay
    except Exception:
        return None, None


def process_new_coordinate(frame_bgr, device):
    global latest_point, new_point_received, tracking, prev_frame, tracking_points, frame_count, failed_frames

    if not new_point_received:
        return

    new_point_received = False
    x, y = latest_point

    if not (0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]):
        return

    # One-shot YOLO segmentation on demand
    results = model.predict(
        frame_bgr,
        device=device,
        verbose=False,
        retina_masks=False,
        imgsz=YOLO_IMGSZ
    )

    mask = get_best_mask_from_point(results, x, y)
    if mask is None:
        return

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != gray.shape[:2]:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))

    params = FEATURE_PARAMS.copy()
    params['mask'] = mask
    corners = cv2.goodFeaturesToTrack(gray, **params)

    if corners is not None:
        tracking_points = corners
        prev_frame = gray.copy()
        tracking = True
        frame_count = 0
        failed_frames = 0


# -----------------------------
# Main
# -----------------------------
def main():
    global serial_port, tracking, prev_frame, tracking_points, frame_count, failed_frames
    global zoom_level, zoom_command, zoom_center, target_selected, display_to_original_coord, model

    # Initialize GStreamer with error handling
    try:
        Gst.init(None)
        print("GStreamer initialized successfully")
    except Exception as e:
        print(f"GStreamer initialization failed: {e}")
        # Continue without GStreamer - we'll use direct OpenCV

    # Serial
    serial_port = setup_serial()

    # YOLO model on CPU (Pi 5); half-precision is not used on CPU.
    model = YOLO("yolo11n-seg.pt")
    device = "cpu"

    # UDP receiver
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()

    # Camera: Try multiple pipeline options for compatibility
    cap_pipelines = gstreamer_camera_pipeline(FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS)
    cap = None
    
    # Try each pipeline until one works
    for i, pipeline in enumerate(cap_pipelines):
        print(f"Trying camera pipeline {i+1}: {pipeline[:50]}...")
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"Successfully opened camera with pipeline {i+1}")
                break
            else:
                cap.release()
                cap = None
        except Exception as e:
            print(f"Pipeline {i+1} failed: {e}")
            if cap:
                cap.release()
                cap = None
    
    # Final fallback: try direct OpenCV camera access
    if cap is None:
        print("Trying direct OpenCV camera access...")
        for device_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    # Set MJPEG codec first (camera's native format)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Try camera's native resolution first
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"Successfully opened camera device {device_id} with direct OpenCV (native resolution)")
                        # Now set desired resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            except Exception as e:
                print(f"Device {device_id} failed: {e}")
                if cap:
                    cap.release()
                    cap = None
    
    if cap is None:
        raise RuntimeError("Failed to open camera with any method. Check camera connections and permissions.")

    # Output pipeline - with error handling
    pipeline = None
    appsrc = None
    try:
        pipeline_str = build_output_pipeline(FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, SINK_HOST, SINK_PORT, BITRATE_KBPS)
        pipeline = Gst.parse_launch(pipeline_str)
        appsrc = pipeline.get_by_name("source")
        # Set caps on appsrc explicitly (more robust negotiation)
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={TARGET_FPS}/1"
        )
        appsrc.set_property("caps", caps)
        pipeline.set_state(Gst.State.PLAYING)
        print("GStreamer output pipeline initialized successfully")
    except Exception as e:
        print(f"GStreamer output pipeline failed: {e}")
        print("Continuing without video streaming output")
        pipeline = None
        appsrc = None

    last_serial_time = 0.0
    serial_interval = 1.0 / float(TARGET_FPS)

    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame_counter += 1
            if frame_counter % (TARGET_FPS * 4) == 0:
                # lightweight periodic log
                print(f"[{time.strftime('%H:%M:%S')}] frames: {frame_counter} tracking={tracking}")

            # Zoom step (event-driven)
            if zoom_command == 'zoom_in' and zoom_level < 3.0:
                zoom_level += 0.5
                zoom_command = None
            elif zoom_command == 'zoom_out' and zoom_level > 1.0:
                zoom_level -= 0.5
                zoom_command = None

            # Process new click/selection (runs YOLO once)
            process_new_coordinate(frame, device)

            # Tracking step (LK optical flow)
            center_x = center_y = 0
            if not target_selected:
                tracking = False
                failed_frames = 0

            if tracking and tracking_points is not None and prev_frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, tracking_points, None, **LK_PARAMS)

                good_new = next_pts[status == 1] if next_pts is not None else np.array([])
                # good_old = tracking_points[status == 1]  # not used for now

                if good_new is not None and len(good_new) > 2:
                    frame_count += 1

                    if frame_count % RESEGMENT_INTERVAL == 0:
                        new_corners, _ = resegment_from_tracking_points(frame, good_new, device)
                        tracking_points = new_corners if new_corners is not None else good_new.reshape(-1, 1, 2)
                    else:
                        tracking_points = good_new.reshape(-1, 1, 2)

                    if len(tracking_points.shape) == 3:
                        center_x = int(np.mean(tracking_points[:, 0, 0]))
                        center_y = int(np.mean(tracking_points[:, 0, 1]))
                    else:
                        center_x = int(np.mean(tracking_points[:, 0]))
                        center_y = int(np.mean(tracking_points[:, 1]))

                    prev_frame = gray.copy()
                    failed_frames = 0
                else:
                    tracking = False
                    tracking_points = None
                    failed_frames += 1

                if failed_frames > max_failed_frames:
                    tracking = False
                    failed_frames = 0

            # Zoomed display mapping (for overlays + back-mapping)
            display_frame = frame
            zoom_x1 = zoom_y1 = 0
            zoom_applied = False

            if zoom_level > 1.0 and zoom_center is not None:
                h, w = display_frame.shape[:2]
                cx, cy = zoom_center
                zw = int(w / zoom_level)
                zh = int(h / zoom_level)

                zoom_x1 = max(0, min(w - zw, int(cx - zw // 2)))
                zoom_y1 = max(0, min(h - zh, int(cy - zh // 2)))

                roi = display_frame[zoom_y1:zoom_y1 + zh, zoom_x1:zoom_x1 + zw]
                display_frame = cv2.resize(roi, (w, h))
                zoom_applied = True

            def original_to_display_coord(ox, oy):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(ox), int(oy)
                rel_x = ox - zoom_x1
                rel_y = oy - zoom_y1
                return int(rel_x * zoom_level), int(rel_y * zoom_level)

            def local_display_to_original_coord(dx, dy):
                if not zoom_applied or zoom_level <= 1.0:
                    return int(dx), int(dy)
                rel_x = dx / zoom_level
                rel_y = dy / zoom_level
                ox = int(rel_x + zoom_x1)
                oy = int(rel_y + zoom_y1)
                h, w = frame.shape[:2]
                return max(0, min(ox, w - 1)), max(0, min(oy, h - 1))

            display_to_original_coord = lambda dx, dy: local_display_to_original_coord(dx, dy)

            # Lightweight overlays (no YOLO plots unless DEBUG_DRAW=True)
            if tracking and tracking_points is not None:
                for pt in tracking_points:
                    # pt is shape (1,2) or (2,)
                    arr = pt.ravel().astype(int)
                    x, y = int(arr[0]), int(arr[1])
                    dx, dy = original_to_display_coord(x, y)
                    cv2.circle(display_frame, (dx, dy), 3, (0, 255, 0), -1)

                cx_d, cy_d = original_to_display_coord(center_x, center_y)
                cv2.circle(display_frame, (cx_d, cy_d), 5, (0, 0, 255), -1)

                # Rate-limited serial
                now = time.time()
                if now - last_serial_time >= serial_interval:
                    fh, fw = frame.shape[:2]
                    send_data_to_serial(center_x, center_y, tracking, fw, fh)
                    last_serial_time = now

            status = f"X={int(center_x)}, Y={int(center_y)}" if tracking else "Tracking: OFF"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if tracking else (0, 0, 255), 2)
            target_status = "Target Selected" if target_selected else "No Target"
            cv2.putText(display_frame, target_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if target_selected else (0, 0, 255), 2)
            cv2.putText(display_frame, f"Zoom: {zoom_level:.1f}x", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Push to GStreamer (appsrc timestamps handled by do-timestamp=true)
            if appsrc is not None:
                try:
                    buf = Gst.Buffer.new_allocate(None, display_frame.nbytes, None)
                    buf.fill(0, display_frame.tobytes())
                    appsrc.emit("push-buffer", buf)
                except Exception as e:
                    print(f"GStreamer buffer push failed: {e}")
                    appsrc = None  # Disable further attempts

            # Optional stop signal (cheap check)
            if os.path.exists("stop.signal"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if appsrc is not None:
                appsrc.emit("end-of-stream")
        except Exception:
            pass
        try:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if serial_port and serial_port.is_open:
                serial_port.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
