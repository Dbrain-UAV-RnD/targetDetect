# GStreamer RTSP 서버 설치 가이드

## 필요한 패키지 설치

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gir1.2-gst-rtsp-server-1.0 \
    libgstreamer1.0-dev \
    v4l-utils
```

### 웹캠 확인
```bash
# 웹캠 장치 확인
v4l2-ctl --list-devices

# 웹캠이 /dev/video0에 없다면 코드에서 device 경로 수정
```

## 실행 방법

### 서버 실행
```bash
python3 webcam_rtsp_server.py
```

### 클라이언트에서 스트림 보기

**VLC 사용:**
```bash
vlc rtsp://localhost:8554/webcam
```

**FFplay 사용:**
```bash
ffplay rtsp://localhost:8554/webcam
```

**GStreamer 사용:**
```bash
gst-launch-1.0 playbin uri=rtsp://localhost:8554/webcam
```

## 문제 해결

### 웹캠이 /dev/video0에 없는 경우
`v4l2-ctl --list-devices` 명령으로 올바른 장치 경로를 확인하고,
`webcam_rtsp_server.py` 파일에서 `device=/dev/video0`를 올바른 경로로 수정하세요.

### 포트가 이미 사용 중인 경우
코드에서 포트 번호를 변경하거나, 다른 프로세스를 종료하세요:
```bash
sudo lsof -i :8554
```
