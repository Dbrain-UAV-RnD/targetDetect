# GStreamer RTSP Server 예제

웹캠 영상을 RTSP 프로토콜로 송출하는 Python 예제입니다.

## 파일 설명

1. **webcam_rtsp_server.py** - 실제 웹캠 영상을 스트리밍
2. **test_pattern_rtsp_server.py** - 테스트 패턴을 스트리밍 (웹캠 없이 테스트 가능)

## 설치

자세한 설치 방법은 [INSTALL.md](INSTALL.md)를 참고하세요.

간단 설치:
```bash
sudo apt-get update
sudo apt-get install -y python3-gi gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gir1.2-gst-rtsp-server-1.0 libgstreamer1.0-dev v4l-utils
```

## 사용 방법

### 테스트 패턴 서버 실행 (웹캠 불필요)
```bash
python3 test_pattern_rtsp_server.py
```

### 웹캠 서버 실행
```bash
python3 webcam_rtsp_server.py
```

### 클라이언트에서 시청

**VLC Player:**
```bash
vlc rtsp://localhost:8554/test     # 테스트 패턴
vlc rtsp://localhost:8554/webcam   # 웹캠
```

**FFplay:**
```bash
ffplay rtsp://localhost:8554/test
```

**GStreamer:**
```bash
gst-launch-1.0 playbin uri=rtsp://localhost:8554/test
```

## WSL2 사용자 주의사항

WSL2 환경에서는 웹캠 접근이 제한적입니다. 다음 중 하나를 선택하세요:

1. **테스트 패턴 사용**: `test_pattern_rtsp_server.py` 실행
2. **Windows에서 실행**: Python과 GStreamer를 Windows에 직접 설치
3. **WSL2 USB 패스스루 설정**: usbipd-win을 사용하여 USB 웹캠을 WSL2에 연결

## 기술 스택

- **GStreamer 1.0**: 멀티미디어 프레임워크
- **gst-rtsp-server**: RTSP 서버 라이브러리
- **Python 3**: PyGObject를 통한 GStreamer 바인딩
- **H.264**: 비디오 코덱
- **RTP**: 실시간 전송 프로토콜

## 파이프라인 설명

### 웹캠 파이프라인
```
v4l2src (웹캠) → videoconvert → x264enc (H.264 인코딩) → rtph264pay (RTP 패킹) → RTSP
```

### 테스트 패턴 파이프라인
```
videotestsrc (테스트 패턴) → videoconvert → x264enc → rtph264pay → RTSP
```

## 커스터마이징

### 포트 변경
```python
server = WebcamRTSPServer(port=9000, mount_point="/mycam")
```

### 웹캠 장치 변경
`webcam_rtsp_server.py` 파일에서:
```python
pipeline = (
    "v4l2src device=/dev/video1 ! "  # video0 → video1로 변경
    ...
)
```

### 해상도/프레임레이트 변경
```python
"video/x-raw,width=1280,height=720,framerate=60/1 ! "
```

### 비트레이트 변경
```python
"x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! "
```
