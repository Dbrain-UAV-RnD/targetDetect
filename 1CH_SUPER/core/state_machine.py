import time
from enum import Enum

from config import (TERMINAL_DEPTH_M, TOF_TERMINAL_M, TOF_CONTACT_M,
                    DEPTH_TOF_DIVERGE_M, REACQ_TIMEOUT_S)


class State(Enum):
    IDLE = 0
    TRACK = 1
    REACQUIRE = 2
    TERMINAL = 3
    CONTACT = 4
    ESTOP = 5


class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self._entered = time.monotonic()
        self.reason = ""

    def _to(self, s, reason=""):
        if s is not self.state:
            self.state = s
            self._entered = time.monotonic()
            self.reason = reason

    @property
    def age(self):
        return time.monotonic() - self._entered

    def on_target_selected(self):
        if self.state in (State.IDLE, State.TRACK, State.REACQUIRE):
            self._to(State.TRACK, "target selected")

    def on_target_cleared(self):
        if self.state is not State.ESTOP:
            self._to(State.IDLE, "target cleared")

    def on_watchdog_trip(self, why):
        self._to(State.ESTOP, why)

    def reset(self):
        self._to(State.IDLE, "reset")

    def step(self, track_ok, tof_m, depth_m, bumper):
        s = self.state
        if s in (State.IDLE, State.ESTOP, State.CONTACT):
            return s

        if bumper or (tof_m is not None and tof_m <= TOF_CONTACT_M):
            self._to(State.CONTACT, f"bumper={bumper} tof={tof_m}")
            return self.state

        if (s is State.TERMINAL and tof_m is not None and depth_m is not None
                and abs(tof_m - depth_m) > DEPTH_TOF_DIVERGE_M):
            self._to(State.ESTOP, f"depth/tof diverge {depth_m:.2f}/{tof_m:.2f}")
            return self.state

        if s is State.TRACK:
            if not track_ok:
                self._to(State.REACQUIRE, "tracker lost")
            elif ((tof_m is not None and tof_m <= TOF_TERMINAL_M) or
                  (depth_m is not None and depth_m <= TERMINAL_DEPTH_M)):
                self._to(State.TERMINAL, f"tof={tof_m} depth={depth_m}")

        elif s is State.REACQUIRE:
            if track_ok:
                self._to(State.TRACK, "reacquired")
            elif self.age > REACQ_TIMEOUT_S:
                self._to(State.IDLE, "reacquire timeout")

        elif s is State.TERMINAL:
            if not track_ok and tof_m is None:
                self._to(State.REACQUIRE, "terminal lost")

        return self.state
