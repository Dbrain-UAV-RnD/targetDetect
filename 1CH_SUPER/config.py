import os
import struct


def _env_int(k, d):
    return int(os.environ.get(k, str(d)))


def _env_float(k, d):
    return float(os.environ.get(k, str(d)))


CAP_W   = _env_int("CAP_W", 1280)
CAP_H   = _env_int("CAP_H", 720)
CAP_FPS = _env_int("CAP_FPS", 30)
CAP_DEV_FPS = _env_int("CAP_DEV_FPS", CAP_FPS)
CAM_INDEX = _env_int("CAM_INDEX", 0)
CAM_CTRLS = os.environ.get(
    "CAM_CTRLS",
    "backlight_compensation=0,brightness=-15,saturation=56,auto_exposure=3")

PROC_W = _env_int("PROC_W", 640)
PROC_H = _env_int("PROC_H", 360)

SP_W, SP_H       = _env_int("SP_W", 640), _env_int("SP_H", 400)
DEPTH_W, DEPTH_H = _env_int("DEPTH_W", 320), _env_int("DEPTH_H", 256)

CAM_HFOV_DEG = _env_float("CAM_HFOV_DEG", 66.0)
CAM_VFOV_DEG = _env_float("CAM_VFOV_DEG", 41.0)

TRACKER = os.environ.get("TRACKER", "nanotrack")
TRACK_BUDGET_MS = _env_float("TRACK_BUDGET_MS", 20.0)
TRACK_CONF_THRESH = _env_float("TRACK_CONF_THRESH", 0.5)
TRACK_APCE_MIN = _env_float("TRACK_APCE_MIN", 0.0)
BOX_MAX_FRAC = _env_float("BOX_MAX_FRAC", 0.5)
TRACK_GRACE_S = _env_float("TRACK_GRACE_S", 0.5)
COLOR_MAX_D = _env_float("COLOR_MAX_D", 0.45)
COLOR0_MAX_D = _env_float("COLOR0_MAX_D", 0.7)
COLOR_EMA_GATE = _env_float("COLOR_EMA_GATE", 0.1)
COLOR_GRID = _env_int("COLOR_GRID", 16)
COLOR_EMA = _env_float("COLOR_EMA", 0.002)
TRACK_LOST_FRAMES = _env_int("TRACK_LOST_FRAMES", 10)
# 근접(박스 점유율 큼) 구간: 수동초점 블러로 score/SP가 무너져도
# 색 게이트만으로 트랙을 유지한다
TERM_HOLD_FRAC   = _env_float("TERM_HOLD_FRAC", 0.12)
TERM_CONF_THRESH = _env_float("TERM_CONF_THRESH", 0.25)
NANOTRACK_DIR = os.environ.get("NANOTRACK_DIR", "/home/gimbal/models/nanotrack")

YAW_KP          = _env_float("YAW_KP", 0.8)
YAW_RATE_MAX    = _env_float("YAW_RATE_MAX", 45.0)
SPEED_MAX       = _env_float("SPEED_MAX", 1.0)
SLOW_DEPTH_M    = _env_float("SLOW_DEPTH_M", 3.0)
TERMINAL_DEPTH_M = _env_float("TERMINAL_DEPTH_M", 1.0)
TOF_TERMINAL_M  = _env_float("TOF_TERMINAL_M", 1.0)
TOF_CONTACT_M   = _env_float("TOF_CONTACT_M", 0.10)
DEPTH_TOF_DIVERGE_M = _env_float("DEPTH_TOF_DIVERGE_M", 1.5)
REACQ_TIMEOUT_S = _env_float("REACQ_TIMEOUT_S", 10.0)

HAILO_RESULT_MAX_AGE_S = _env_float("HAILO_RESULT_MAX_AGE_S", 0.5)

STAB_ENABLE  = os.environ.get("STAB", "1") not in ("0", "", "false", "no")
STAB_W       = _env_int("STAB_W", 240)
STAB_H       = _env_int("STAB_H", 135)
STAB_MARGIN  = _env_float("STAB_MARGIN", 0.05)
STAB_TAU     = _env_float("STAB_TAU", 0.4)
STAB_TAU_MIN = _env_float("STAB_TAU_MIN", 0.1)
STAB_TAU_MAX = _env_float("STAB_TAU_MAX", 2.0)
STAB_CORNERS = _env_int("STAB_CORNERS", 40)
STAB_MIN_PTS = _env_int("STAB_MIN_PTS", 12)

TERM_LOCK_FRAC      = _env_float("TERM_LOCK_FRAC", 0.35)
TERM_LOCK_TIMEOUT_S = _env_float("TERM_LOCK_TIMEOUT_S", 3.0)
COAST_S             = _env_float("COAST_S", 0.4)
SLEW_ANG_DEG_S      = _env_float("SLEW_ANG_DEG_S", 90.0)
SLEW_N_S            = _env_float("SLEW_N_S", 3.0)

WDG_FRAME_TIMEOUT_S = _env_float("WDG_FRAME_TIMEOUT_S", 0.1)
WDG_TEMP_LOG_S      = _env_float("WDG_TEMP_LOG_S", 5.0)

FAST_LOOP_CORES = {int(c) for c in os.environ.get("FAST_CORES", "0,1").split(",")}
HAILO_CORES     = {int(c) for c in os.environ.get("HAILO_CORES", "2").split(",")}

SHM_FRAME  = "1chs_frame"
SHM_RESULT = "1chs_result"
SHM_CTRL   = "1chs_ctrl"

ANCHOR_MAX_KP = _env_int("ANCHOR_MAX_KP", 200)
SP_CONF_THRESH = _env_float("SP_CONF_THRESH", 0.015)
SP_NMS_RADIUS  = _env_int("SP_NMS_RADIUS", 4)
REACQ_MIN_MATCHES = _env_int("REACQ_MIN_MATCHES", 12)
AUDIT_EVERY = _env_int("AUDIT_EVERY", 10)
AUDIT_FAILS = _env_int("AUDIT_FAILS", 5)
AUDIT_REFRESH = _env_int("AUDIT_REFRESH", 10)

HEF_SUPERPOINT = os.environ.get("HEF_SUPERPOINT", "/home/gimbal/models/superpoint.hef")
HEF_DEPTH      = os.environ.get("HEF_DEPTH", "/home/gimbal/models/scdepthv3.hef")

LOG_DIR = os.environ.get("LOG_DIR", "/home/gimbal/1CH_SUPER/logs")

RTSP_ENABLE  = os.environ.get("RTSP", "1") not in ("0", "", "false")
RTSP_PORT    = _env_int("RTSP_PORT", 554)
RTSP_PATH    = os.environ.get("RTSP_PATH", "/video0")
RTSP_W       = _env_int("RTSP_W", 1920)
RTSP_H       = _env_int("RTSP_H", 1080)
RTSP_FPS     = _env_int("RTSP_FPS", 12)
RTSP_BITRATE = _env_int("RTSP_BITRATE", 2500)
RTSP_PRESET  = os.environ.get("RTSP_PRESET", "veryfast")
RTSP_CODEC   = os.environ.get("RTSP_CODEC", "h264").lower()
RTSP_GOP     = _env_int("RTSP_GOP", RTSP_FPS * 2)
RTSP_VBV     = _env_int("RTSP_VBV", 300)
RTSP_QUEUE   = _env_int("RTSP_QUEUE", 3)
RTSP_INTRA_REFRESH = os.environ.get("RTSP_INTRA_REFRESH", "1") not in ("0", "false", "no")
OSD_BOX_STICKY_FRAC = _env_float("OSD_BOX_STICKY_FRAC", 0.06)
RTSP_X265_OPTS = os.environ.get(
    "RTSP_X265_OPTS",
    "no-rect=1:no-amp=1:wpp=1:pmode=1:pme=1:frame-threads=4:rd=1:me=0:subme=0")


GCS_UDP_PORT = _env_int("GCS_UDP_PORT", 37260)
GCS_HEADER1  = 0x55
GCS_HEADER2  = 0x66
GCS_CMD_OFFSET     = 7
GCS_PAYLOAD_OFFSET = 8

CMD_CAM_HEARTBEAT   = 0
CMD_AI_MODE         = 4
CMD_GIMBAL_ZOOM     = 5
CMD_TRACK_ACTION    = 6
CMD_GIMBAL_ROTATE   = 7
CMD_GIMBAL_CENTER   = 8
CMD_SET_GAIN        = 10
CMD_SET_OSD_DISPLAY = 13
CMD_TEST_DIGITAL_ZOOM = 20
CMD_TEST_ZOOM_RAW     = 22
CMD_STABILIZER_MODE   = 31
CMD_STABILIZER_ALPHA  = 32
GCS_STAB_RESET        = 0xFF

GCS_REF_W, GCS_REF_H = 1920, 1080
GCS_ZOOM_RAW_MAX = 0x4000
MAX_ZOOM     = _env_float("MAX_ZOOM", 5.0)
ZOOM_RATE    = _env_float("GCS_ZOOM_RATE", 2.0)
ZOOM_TIMEOUT = _env_float("GCS_ZOOM_TIMEOUT", 3.0)
DEADBAND_FRAC = _env_float("DEADBAND_FRAC", 0.03)
FOLLOW_TAU     = _env_float("FOLLOW_TAU", 0.065)
FOLLOW_TAU_OFF = _env_float("FOLLOW_TAU_OFF", 0.205)

FCC_TX_HEADER1, FCC_TX_HEADER2 = 0xBB, 0x88
FCC_RX_HEADER1, FCC_RX_HEADER2 = 0xBB, 0x99
FCC_TX_FMT  = "<BBBffBffbBhhhhhh10bB"
FCC_TX_SIZE = struct.calcsize(FCC_TX_FMT)
FCC_RX_FMT  = "<BBfBffddffffffff32sB"
FCC_RX_SIZE = struct.calcsize(FCC_RX_FMT)
assert FCC_TX_SIZE == 45, FCC_TX_SIZE
assert FCC_RX_SIZE == 96, FCC_RX_SIZE

FCC_PORT  = os.environ.get("FCC_PORT", "/dev/ttyAMA3")
FCC_BAUD  = _env_int("FCC_BAUD", 115200)
FCC_HZ    = _env_float("FCC_HZ", 50)
FCC_RETRY = _env_float("FCC_RETRY", 2.0)

TOF_I2C_BUS  = _env_int("TOF_I2C_BUS", 1)
TOF_I2C_ADDR = _env_int("TOF_I2C_ADDR", 0x29)
BUMPER_GPIO  = _env_int("BUMPER_GPIO", 17)
