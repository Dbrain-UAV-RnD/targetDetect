import cv2
import torch
from ultralytics import SAM, FastSAM, YOLO
import numpy as np

model = SAM("mobile_sam.pt")
# model = FastSAM("FastSAM-s.pt")
# model = YOLO("yolo11n-seg.pt")
device = 0 if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

if not cap.isOpened():
    raise RuntimeError("not opend camera")

segmented_frame = None
current_frame = None
prev_frame = None
tracking_points = None
tracking = False
frame_count = 0
resegment_interval = 5

lk_params = dict(winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def resegment_from_tracking_points(frame, points):
    if len(points) < 3:
        return None, None
    
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

def mouse_callback(event, x, y, flags, param):
    global segmented_frame, current_frame, tracking_points, tracking, prev_frame, frame_count
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
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
            print(f"KLT 추적 시작: {len(tracking_points)}개 특징점")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    current_frame = frame.copy()
    display_frame = frame.copy()
    
    if tracking and tracking_points is not None and prev_frame is not None:
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, tracking_points, None, **lk_params)
        
        good_new = next_pts[status == 1]
        good_old = tracking_points[status == 1]
        
        if len(good_new) > 2:
            frame_count += 1
            
            if frame_count % resegment_interval == 0:
                new_corners, new_segmented = resegment_from_tracking_points(current_frame, good_new)
                if new_corners is not None:
                    tracking_points = new_corners
                    segmented_frame = new_segmented
                    print(f"재세그멘테이션: {len(tracking_points)}개 특징점 업데이트")
                else:
                    tracking_points = good_new.reshape(-1, 1, 2)
            else:
                tracking_points = good_new.reshape(-1, 1, 2)
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv2.circle(display_frame, (a, b), 3, (0, 255, 0), -1)
                cv2.line(display_frame, (a, b), (c, d), (0, 255, 0), 1)
            
            prev_frame = gray_frame.copy()
        else:
            tracking = False
            tracking_points = None
            print("추적 중단: 충분한 특징점이 없음")
    
    if not tracking:
        display_frame = current_frame
    
    cv2.imshow("MobileSAM with KLT", display_frame)
    cv2.setMouseCallback("MobileSAM with KLT", mouse_callback)
    
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('c'):
        segmented_frame = None
        tracking = False
        tracking_points = None
        frame_count = 0
        print("추적 초기화")
    elif key == ord('s') and tracking:
        tracking = False
        tracking_points = None
        frame_count = 0
        print("추적 중단")
    elif key == ord('r') and tracking:
        new_corners, new_segmented = resegment_from_tracking_points(current_frame, tracking_points)
        if new_corners is not None:
            tracking_points = new_corners
            segmented_frame = new_segmented
            prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            print(f"수동 재세그멘테이션: {len(tracking_points)}개 특징점")

cap.release()
cv2.destroyAllWindows()
