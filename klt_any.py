import cv2
import numpy as np

# ===== 파라미터 =====
ROI_SIZE = 100
MIN_POINTS = 3
MAX_CORNERS = 30
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
GFTT_BLOCKSIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# ROI 이동 관련 파라미터
ROI_MOVE_ALPHA = 1  # ROI 이동 smoothing factor (0~1, 작을수록 부드럽게)
MIN_POINTS_FOR_MOVE = 0  # ROI 이동을 위한 최소 포인트 수

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. /dev/video0 확인")

prev_gray = None
p0 = None                  # (N,1,2) float32
roi_rect = None            # (x,y,w,h)
roi_center = None          # ROI 중심점 (부드러운 이동용)
adaptive_mode = True       # ROI 자동 이동 모드
win_name = "Adaptive ROI Tracker (click to set ROI)"
cv2.namedWindow(win_name)

def clamp_roi(cx, cy, w, h, W, H):
    x = int(cx - w // 2)
    y = int(cy - h // 2)
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return x, y, w, h

def detect_points(gray, rect):
    x, y, w, h = rect
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE, mask=mask, blockSize=GFTT_BLOCKSIZE,
        useHarrisDetector=False
    )
    return pts

def in_roi(pts, rect):
    if pts is None or len(pts) == 0:
        return np.array([], dtype=bool)
    x, y, w, h = rect
    xs = pts[:, 0]
    ys = pts[:, 1]
    return (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)

def calculate_centroid(pts):
    """포인트들의 중심점 계산"""
    if pts is None or len(pts) == 0:
        return None
    return np.mean(pts.reshape(-1, 2), axis=0)

def update_roi_position(roi_rect, target_center, W, H, alpha=0.15):
    """ROI를 타겟 중심점으로 부드럽게 이동"""
    x, y, w, h = roi_rect
    current_cx = x + w // 2
    current_cy = y + h // 2
    
    # 부드러운 이동 (exponential moving average)
    new_cx = current_cx + alpha * (target_center[0] - current_cx)
    new_cy = current_cy + alpha * (target_center[1] - current_cy)
    
    # 경계 체크와 함께 새로운 ROI 위치 반환
    return clamp_roi(new_cx, new_cy, w, h, W, H)

def on_mouse(event, x, y, flags, userdata):
    global roi_rect, roi_center, p0, prev_gray
    if event == cv2.EVENT_LBUTTONDOWN and prev_gray is not None:
        H, W = prev_gray.shape[:2]
        roi_rect = clamp_roi(x, y, ROI_SIZE, ROI_SIZE, W, H)
        roi_center = (x, y)  # 초기 ROI 중심점 설정
        p0 = detect_points(prev_gray, roi_rect)

cv2.setMouseCallback(win_name, on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    if prev_gray is None:
        prev_gray = gray.copy()

    # ROI가 설정되었고 포인트가 없으면 ROI 내부에서 재검출
    if roi_rect is not None and (p0 is None or len(p0) == 0):
        p0 = detect_points(gray, roi_rect)

    # 추적
    if roi_rect is not None and p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA
        )
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Adaptive 모드: 포인트들의 중심으로 ROI 이동
            if adaptive_mode and len(good_new) >= MIN_POINTS_FOR_MOVE:
                centroid = calculate_centroid(good_new)
                if centroid is not None:
                    # ROI를 포인트 중심으로 부드럽게 이동
                    roi_rect = update_roi_position(roi_rect, centroid, W, H, ROI_MOVE_ALPHA)
                    
                    # 이동한 ROI 기준으로 포인트 필터링
                    mask_in = in_roi(good_new, roi_rect)
                    good_new = good_new[mask_in]
                    good_old = good_old[mask_in]
            else:
                # Fixed 모드: 기존처럼 ROI 밖 포인트만 제거
                mask_in = in_roi(good_new, roi_rect)
                good_new = good_new[mask_in]
                good_old = good_old[mask_in]

            # 트랙/점 시각화
            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv2.line(frame, (a, b), (c, d), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(frame, (a, b), 2, (0, 0, 255), -1, cv2.LINE_AA)

            # 포인트 중심점 표시 (디버깅용)
            if len(good_new) > 0:
                centroid = calculate_centroid(good_new)
                if centroid is not None:
                    cv2.circle(frame, tuple(centroid.astype(int)), 5, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(frame, "C", tuple(centroid.astype(int) + np.array([8, 4])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

            # 다음 프레임용 업데이트
            if len(good_new) > 0:
                p0 = good_new.reshape(-1, 1, 2).astype(np.float32)
            else:
                p0 = None  # 전부 탈락

            # 포인트가 부족하면 현재 ROI 내부에서 재검출
            if (p0 is None or len(p0) < MIN_POINTS) and roi_rect is not None:
                p0 = detect_points(gray, roi_rect)

    # ROI 그리기 (adaptive 모드에 따라 색상 변경)
    if roi_rect is not None:
        x, y, w, h = roi_rect
        color = (0, 255, 255) if adaptive_mode else (255, 128, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
        
        # ROI 중심점 표시
        cx, cy = x + w//2, y + h//2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

    # 안내 텍스트
    mode_text = "ADAPTIVE" if adaptive_mode else "FIXED"
    cv2.putText(frame, f"Mode: {mode_text} | Click: set ROI | a: toggle mode | r: reset | ESC: quit",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    npts = 0 if p0 is None else len(p0)
    cv2.putText(frame, f"Tracked points: {npts} | Min for move: {MIN_POINTS_FOR_MOVE}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
    
    # 추가 정보 표시
    if roi_rect is not None:
        x, y, w, h = roi_rect
        cv2.putText(frame, f"ROI: ({x},{y}) {w}x{h}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset
        p0 = None
        roi_rect = None
        roi_center = None
    elif key == ord('a'):  # Toggle adaptive mode
        adaptive_mode = not adaptive_mode
        print(f"Adaptive mode: {adaptive_mode}")

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()