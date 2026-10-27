"""RoboTHOR ObjectNav (2021 challenge protocol) on AI2-THOR 5.0.

Environment settings follow robothor-challenge/challenge_config.yaml (LoCoBot, stochastic actuation, 0.25 m
moves, 30-degree turns, 640x480, vertical FOV 63.45, visibilityDistance 1.0, 500 steps). An episode succeeds if
the agent issues Stop while an object of the target type is visible (AI2-THOR's visibility test with
visibilityDistance = 1 m), as in the challenge. SPL uses the dataset's shortest_path_length.
"""
import gzip
import json
import math
import os

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "robothor")
CONFIG = dict(agentMode="locobot", agentType="stochastic", visibilityDistance=1.0, gridSize=0.25,
              rotateStepDegrees=30, continuousMode=True, snapToGrid=False, fieldOfView=63.453048374758716,
              width=640, height=480, renderDepthImage=True)
ACTIONS = ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown", "Stop"]
MAX_STEPS = 500


def load_split(split="val"):
    eps = []
    d = os.path.join(DATA, split)
    for f in sorted(os.listdir(d)):
        if f.endswith(".json.gz"):
            eps += json.load(gzip.open(os.path.join(d, f)))
    return eps


class RoboTHOREnv:
    def __init__(self, platform=CloudRendering):
        self.c = Controller(platform=platform, **CONFIG)
        self.scene = None

    def reset(self, ep):
        if ep["scene"] != self.scene:
            self.c.reset(scene=ep["scene"])
            self.scene = ep["scene"]
        rot = {"x": 0, "y": ep["initial_orientation"], "z": 0}
        try:
            ev = self.c.step(action="TeleportFull", position=ep["initial_position"], rotation=rot,
                             horizon=ep["initial_horizon"], standing=True)
        except Exception:
            ev = self.c.step(action="TeleportFull", position=ep["initial_position"], rotation=rot,
                             horizon=ep["initial_horizon"])
        if not ev.metadata["lastActionSuccess"]:
            raise RuntimeError(f"teleport failed in {ep['scene']}: {ev.metadata['errorMessage'][:120]}")
        self.ep, self.steps, self.path_len, self.done = ep, 0, 0.0, False
        self.success = False
        # camera height above the floor is a fixed property of the robot; the dataset's path points lie on the
        # floor, so the first one gives the floor level at the start (equivalently: a one-time calibration)
        floor_y = ep["shortest_path"][0]["y"] if ep.get("shortest_path") else ev.metadata["agent"]["position"]["y"] - 0.9
        self.cam_h = ev.metadata["cameraPosition"]["y"] - floor_y
        return self.obs(ev)

    def obs(self, ev=None):
        ev = ev or self.c.last_event
        a = ev.metadata["agent"]
        return {"rgb": ev.frame, "depth": ev.depth_frame, "step": self.steps, "cam_h": self.cam_h,
                "pose": {"x": a["position"]["x"], "z": a["position"]["z"], "yaw": a["rotation"]["y"],
                         "horizon": a["cameraHorizon"]}}

    def step(self, action):
        assert not self.done
        prev = self.c.last_event.metadata["agent"]["position"]
        if action == "Stop":
            ev = self.c.last_event
            self.success = self.target_visible(ev)
            self.done = True
        else:
            ev = self.c.step(action=action)
            cur = ev.metadata["agent"]["position"]
            self.path_len += math.hypot(cur["x"] - prev["x"], cur["z"] - prev["z"])
        self.steps += 1
        if self.steps >= MAX_STEPS:
            self.done = True
        info = {"ok": bool(ev.metadata["lastActionSuccess"]), "action": action}
        return self.obs(ev), self.done, info

    def target_visible(self, ev=None):
        ev = ev or self.c.last_event
        return any(o["visible"] and o["objectType"] == self.ep["object_type"] for o in ev.metadata["objects"])

    def target_distance(self, ev=None):
        """Analysis only: 2D distance to the nearest target-type object."""
        ev = ev or self.c.last_event
        a = ev.metadata["agent"]["position"]
        ds = [math.hypot(o["position"]["x"] - a["x"], o["position"]["z"] - a["z"])
              for o in ev.metadata["objects"] if o["objectType"] == self.ep["object_type"]]
        return min(ds) if ds else None

    def spl(self):
        if not self.success:
            return 0.0
        L = self.ep.get("shortest_path_length") or 0.0
        return L / max(L, self.path_len) if L > 0 else 1.0

    def close(self):
        self.c.stop()
