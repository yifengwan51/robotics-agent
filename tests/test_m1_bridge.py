"""M1 bridge tests. Everything runs on the loopback interface and a throwaway DDS domain (77),
so nothing can reach the real robot (which listens on domain 0 over its Ethernet port)."""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embagent.robot import messages as m  # noqa: E402
from embagent.robot.m1_bridge import M1Bridge, SafetyLimits  # noqa: E402

TEST_DOMAIN = 77

READER = textwrap.dedent("""
    import json, sys, time
    sys.path.insert(0, {root!r})
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from embagent.robot import messages as m
    from embagent.robot.m1_bridge import cyclonedds_config
    dom = Domain({domain}, cyclonedds_config("lo"))  # keep a reference or the config is dropped
    dp = DomainParticipant({domain})
    vel = DataReader(dp, Topic(dp, m.TOPIC_VELCMD, m.VelCmd_))
    mode = DataReader(dp, Topic(dp, m.TOPIC_SPORTMODECMD, m.SportModeCmd_))
    print("READY", flush=True)
    end = time.time() + {seconds}
    while time.time() < end:
        for s in vel.take(N=100):
            print(json.dumps({{"t": time.time(), "vel": [s.lin_vx, s.lin_vy, s.ang_wz]}}), flush=True)
        for s in mode.take(N=100):
            print(json.dumps({{"t": time.time(), "mode": [s.device_id, s.mode]}}), flush=True)
        time.sleep(0.01)
""")


def _start_reader(seconds):
    code = READER.format(root=ROOT, domain=TEST_DOMAIN, seconds=seconds)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "READY"
    return proc


def _collect(proc):
    out, _ = proc.communicate(timeout=30)
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_dry_run_clamps_and_decays_to_zero():
    limits = SafetyLimits(cmd_timeout=0.2, zero_hold=0.3)
    with M1Bridge(dry_run=True, limits=limits) as b:
        assert b.set_velocity(5.0, -5.0, 9.0) == (0.3, -0.2, 0.5)
        time.sleep(0.15)
        assert b.sent[-1][2] == (0.3, -0.2, 0.5)
        time.sleep(0.25)  # setpoint is stale -> zeros (t~0.40 s, inside zero_hold)
        assert b.sent[-1][2] == (0.0, 0.0, 0.0)
        time.sleep(0.25)  # t~0.65 s > cmd_timeout + zero_hold = 0.5 s -> quiet
        n = len(b.sent)
        time.sleep(0.3)
        assert len(b.sent) == n


def test_unsafe_mode_rejected():
    with M1Bridge(dry_run=True) as b:
        with pytest.raises(ValueError):
            b.set_mode(m.MODE_PASSIVE)
        assert b.emergency_damp()
        assert b.sent[-1][1:] == ("mode", m.MODE_DAMPING)


def test_disarmed_live_bridge_publishes_only_zeros(monkeypatch):
    monkeypatch.delenv("LTR_M1_ARMED", raising=False)
    proc = _start_reader(seconds=2.5)
    with M1Bridge(interface="lo", domain_id=TEST_DOMAIN, dry_run=False, armed=True) as b:
        assert not b.armed
        time.sleep(0.8)  # discovery
        for _ in range(10):
            b.set_velocity(0.2, 0.0, 0.3)
            time.sleep(0.05)
        assert b.set_mode(m.MODE_NORMAL_WALK) is False
    vels = [r["vel"] for r in _collect(proc) if "vel" in r]
    assert vels, "no samples received over loopback"
    assert all(v == [0.0, 0.0, 0.0] for v in vels)


def test_armed_loopback_roundtrip(monkeypatch):
    monkeypatch.setenv("LTR_M1_ARMED", "1")
    proc = _start_reader(seconds=3.0)
    limits = SafetyLimits(cmd_timeout=0.2, zero_hold=0.5)
    with M1Bridge(interface="lo", domain_id=TEST_DOMAIN, dry_run=False, armed=True, limits=limits) as b:
        assert b.armed
        time.sleep(0.8)
        assert b.set_mode(m.MODE_STANDUP)
        for _ in range(10):
            b.set_velocity(1.0, 0.0, -2.0)  # clamped to (0.3, 0, -0.5)
            time.sleep(0.05)
        time.sleep(0.8)  # let the watchdog zero it
    rows = _collect(proc)
    vels = [r["vel"] for r in rows if "vel" in r]
    modes = [r["mode"] for r in rows if "mode" in r]
    assert [m.DEVICE_ID_HIGHLEVELSDK, m.MODE_STANDUP] in modes
    moving = [v for v in vels if any(v)]
    assert moving and all(v[0] == pytest.approx(0.3) and v[2] == pytest.approx(-0.5) for v in moving)
    assert vels[-1] == [0.0, 0.0, 0.0]
