#!/usr/bin/env python3
"""
GStreamer RTSP Server - Webcam Streaming Example
웹캠 영상을 RTSP 프로토콜로 송출하는 간단한 예제
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

class WebcamRTSPServer:
    def __init__(self, port=8554, mount_point="/webcam"):
        """
        RTSP 서버 초기화

        Args:
            port: RTSP 서버 포트 (기본값: 8554)
            mount_point: 스트림 마운트 경로 (기본값: /webcam)
        """
        self.port = port
        self.mount_point = mount_point

        # GStreamer 초기화
        Gst.init(None)

        # RTSP 서버 생성
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))

        # 미디어 팩토리 생성 및 설정
        self.factory = GstRtspServer.RTSPMediaFactory()

        # 웹캠 파이프라인 설정
        # v4l2src: 리눅스 웹캠 소스
        # videoconvert: 비디오 형식 변환
        # x264enc: H.264 인코딩
        # rtph264pay: RTP 페이로드 패킹
        pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            "rtph264pay name=pay0 pt=96"
        )

        self.factory.set_launch(pipeline)
        self.factory.set_shared(True)

        # 마운트 포인트에 팩토리 연결
        mounts = self.server.get_mount_points()
        mounts.add_factory(mount_point, self.factory)

        # 서버를 GLib 메인 컨텍스트에 연결
        self.server.attach(None)

    def run(self):
        """서버 실행"""
        print(f"RTSP 서버 시작됨")
        print(f"스트림 URL: rtsp://localhost:{self.port}{self.mount_point}")
        print(f"\n테스트 방법:")
        print(f"  VLC: vlc rtsp://localhost:{self.port}{self.mount_point}")
        print(f"  FFplay: ffplay rtsp://localhost:{self.port}{self.mount_point}")
        print(f"  GStreamer: gst-launch-1.0 playbin uri=rtsp://localhost:{self.port}{self.mount_point}")
        print(f"\n종료하려면 Ctrl+C를 누르세요")

        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n서버 종료 중...")
            loop.quit()

def main():
    # RTSP 서버 생성 및 실행
    server = WebcamRTSPServer(port=8554, mount_point="/webcam")
    server.run()

if __name__ == "__main__":
    main()
