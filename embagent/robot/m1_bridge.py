"""Safe high-level velocity bridge to the LTR M1 over DDS (CycloneDDS 0.10.x, same as the vendor SDK).

Only uses the two interfaces the vendor documents: rt/ltr/vel_cmd and rt/ltr/sportmode_cmd.
The robot must already be in the vendor's HIGHLEVEL state (G20 remote: "RL Loco Classic", then HOME).

Safety model:
- dry_run=True (default) never touches the network; commands are only recorded.
- Non-zero motion is published only if armed=True AND the env var LTR_M1_ARMED=1 is set.
- Velocities are clamped to SafetyLimits. A background thread republishes the current setpoint while it is
  fresh; once it is older than cmd_timeout it publishes zeros for zero_hold seconds, then goes quiet.
- close() always sends zeros. emergency_damp() is always allowed (vendor DAMPING mode).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from . import messages as m

log = logging.getLogger(__name__)

ARM_ENV = "LTR_M1_ARMED"
_ZERO = (0.0, 0.0, 0.0)
_SAFE_MODES = {m.MODE_DAMPING, m.MODE_SITDOWN, m.MODE_STANDUP, m.MODE_NORMAL_WALK}
_MOTION_MODES = {m.MODE_STANDUP, m.MODE_NORMAL_WALK}


@dataclass
class SafetyLimits:
    max_vx: float = 0.3       # m/s, forward/backward
    max_vy: float = 0.2       # m/s, lateral
    max_wz: float = 0.5       # rad/s, yaw rate
    cmd_timeout: float = 0.3  # s; older setpoints are replaced by zero
    zero_hold: float = 1.0    # s of zero commands after a setpoint goes stale
    rate_hz: float = 20.0


def cyclonedds_config(interface: str) -> str:
    """Bind CycloneDDS to one network interface (the cable to the robot, or 'lo' for tests).

    Loopback is not multicast-capable on Linux, so 'lo' uses unicast discovery against 127.0.0.1.
    """
    discovery = ""
    if interface == "lo":
        discovery = ('<Discovery><Peers><Peer address="127.0.0.1"/></Peers>'
                     "<ParticipantIndex>auto</ParticipantIndex>"
                     "<MaxAutoParticipantIndex>50</MaxAutoParticipantIndex></Discovery>")
    return (
        '<CycloneDDS><Domain Id="any"><General><Interfaces>'
        f'<NetworkInterface name="{interface}" priority="default" multicast="default"/>'
        f"</Interfaces></General>{discovery}</Domain></CycloneDDS>"
    )


_domains = {}


def _participant(domain_id: int, interface: str):
    from cyclonedds.domain import Domain, DomainParticipant

    if domain_id not in _domains:
        _domains[domain_id] = Domain(domain_id, cyclonedds_config(interface))
    return DomainParticipant(domain_id)


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, float(v)))


class M1Bridge:
    def __init__(self, interface: str = "lo", domain_id: int = 0, limits: SafetyLimits | None = None,
                 dry_run: bool = True, armed: bool = False):
        self.limits = limits or SafetyLimits()
        self.dry_run = dry_run
        self.armed = bool(armed and not dry_run and os.environ.get(ARM_ENV) == "1")
        if armed and not dry_run and not self.armed:
            log.warning("armed=True ignored: set %s=1 to allow motion commands", ARM_ENV)
        self.sent = deque(maxlen=10000)  # (t, kind, payload) of everything published or simulated
        self._lock = threading.Lock()
        self._setpoint = _ZERO
        self._stamp = -1e9
        self._warned_disarmed = False

        self._vel_writer = self._mode_writer = None
        if not dry_run:
            from cyclonedds.pub import DataWriter
            from cyclonedds.topic import Topic

            self._dp = _participant(domain_id, interface)
            self._vel_writer = DataWriter(self._dp, Topic(self._dp, m.TOPIC_VELCMD, m.VelCmd_))
            self._mode_writer = DataWriter(self._dp, Topic(self._dp, m.TOPIC_SPORTMODECMD, m.SportModeCmd_))

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="m1-vel", daemon=True)
        self._thread.start()

    # ---- public API -------------------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> tuple:
        """Update the setpoint; it must be refreshed faster than cmd_timeout or it decays to zero."""
        lim = self.limits
        cmd = (_clamp(vx, lim.max_vx), _clamp(vy, lim.max_vy), _clamp(wz, lim.max_wz))
        with self._lock:
            self._setpoint, self._stamp = cmd, time.monotonic()
        return cmd

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)

    def move_for(self, vx: float, vy: float, wz: float, duration: float) -> None:
        """Blocking open-loop motion: hold a velocity for `duration` seconds, then stop."""
        period = 1.0 / self.limits.rate_hz
        end = time.monotonic() + max(0.0, duration)
        while time.monotonic() < end:
            self.set_velocity(vx, vy, wz)
            time.sleep(period)
        self.stop()

    def set_mode(self, mode: int, allow_unsafe: bool = False) -> bool:
        if mode not in _SAFE_MODES and not allow_unsafe:
            raise ValueError(f"mode {mode} not in the safe set {sorted(_SAFE_MODES)}")
        if mode in _MOTION_MODES and not self.armed and not self.dry_run:
            log.warning("mode %d makes the robot move; refused while disarmed", mode)
            return False
        return self._publish_mode(mode)

    def emergency_damp(self) -> bool:
        """Vendor DAMPING mode; always allowed. The robot lowers itself, so use it on flat ground."""
        self.stop()
        return self._publish_mode(m.MODE_DAMPING)

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=1.0)
        for _ in range(5):
            self._publish_vel(_ZERO)
            time.sleep(0.02)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- internals ----------------------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.limits.rate_hz
        while self._running:
            with self._lock:
                age = time.monotonic() - self._stamp
                cmd = self._setpoint
            if age <= self.limits.cmd_timeout:
                self._publish_vel(cmd)
            elif age <= self.limits.cmd_timeout + self.limits.zero_hold:
                self._publish_vel(_ZERO)
            time.sleep(period)

    def _publish_vel(self, cmd: tuple) -> None:
        if any(cmd) and not self.armed and not self.dry_run:
            if not self._warned_disarmed:
                log.warning("disarmed: publishing zero instead of %s", cmd)
                self._warned_disarmed = True
            cmd = _ZERO
        self.sent.append((time.monotonic(), "vel", cmd))
        if self._vel_writer is not None:
            vx, vy, wz = cmd
            self._vel_writer.write(m.VelCmd_(lin_vx=vx, lin_vy=vy, ang_wz=wz))

    def _publish_mode(self, mode: int) -> bool:
        self.sent.append((time.monotonic(), "mode", mode))
        if self._mode_writer is not None:
            self._mode_writer.write(m.SportModeCmd_(device_id=m.DEVICE_ID_HIGHLEVELSDK, mode=mode))
        return True
