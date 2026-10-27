"""S2M for RoboTHOR ObjectNav: forward-only LoCoBot, 30-degree turns, explicit Stop, multi-room apartments.

Reuses S2MPlanner's perception and 3D evidence fusion unchanged. Differences from EB-Navigation:
- motion: heading-lattice A* (MoveAhead / RotateLeft / RotateRight), no strafing;
- termination: the agent must decide to Stop. It stops only after it has come within reach of a committed
  hypothesis, turned and tilted to bring it to the image centre, and re-confirmed it with a presence query;
  a failed confirmation marks the hypothesis exhausted (no reliance on simulator auto-termination);
- search: 360-degree look-around, then frontier exploration over the depth map.
"""
import math

import numpy as np

from embagent.geometry import OccupancyMap, fwd_right, plan_heading, world_to_pixel
from embagent.planners.s2m import S2MPlanner

ROT = 30


class HeadingCore:
    """Map + planner for forward-only robots with continuous (noisy) headings."""

    def __init__(self, step=0.25, rot=ROT, goal_tol=0.55, inflate=0.18, fov=63.453048374758716):
        self.step, self.rot, self.goal_tol, self.inflate, self.fov = step, rot, goal_tol, inflate, fov
        self.map = None
        self.plan, self.plan_target, self.resid, self._expect = [], None, None, None
        self._step = -1

    def reset_state(self):
        self.map = None
        self.plan, self.plan_target, self.resid, self._expect = [], None, None, None
        self._step = -1

    def ensure(self, obs):
        if self.map is None:
            p = obs["pose"]
            self.map = OccupancyMap((p["x"], p["z"]), half_extent=12.0, fov=self.fov, size=obs["depth"].shape[0],
                                    cam_h=obs["cam_h"])
        if obs["step"] == self._step:
            return
        p = obs["pose"]
        self.map.mark_free_disk(p["x"], p["z"], r=0.18)
        self.map.integrate(obs["depth"], p, max_h=obs["cam_h"] + 0.3)
        self._step = obs["step"]

    def record_move(self, pose_before, action, ok):
        if action == "MoveAhead" and not ok:
            fv, _ = fwd_right(pose_before["yaw"])
            a = np.array([pose_before["x"], pose_before["z"]])
            self.map.mark_blocked_segment(a, a + self.step * fv)
            self.map.add_failed_move(a, fv)
            self.plan = []

    def bearing(self, pose, target):
        fv, rv = fwd_right(pose["yaw"])
        rel = np.array([target[0] - pose["x"], target[1] - pose["z"]])
        return math.degrees(math.atan2(rel @ rv, rel @ fv))

    def next_move(self, pose, target, budget=60):
        pos = np.array([pose["x"], pose["z"]])
        target = np.asarray(target, float)
        stale = not self.plan or self.plan_target is None or np.linalg.norm(self.plan_target - target) > 0.3
        if not stale and self._expect is not None:
            ep, ey = self._expect
            dyaw = abs((pose["yaw"] - ey + 180) % 360 - 180)
            if np.linalg.norm(ep - pos) > 0.15 or dyaw > 10:
                stale = True
        if stale:
            acts, states, resid = plan_heading(self.map, pos, pose["yaw"], target, step=self.step, rot=self.rot,
                                               max_cost=min(budget, 80), goal_tol=self.goal_tol, inflate=self.inflate)
            self.plan = list(zip(acts, states))
            self.plan_target, self.resid = target, resid
        if not self.plan:
            self._expect = None
            return None
        a, (p_next, y_next) = self.plan.pop(0)
        self._expect = (np.asarray(p_next), y_next)
        return {"F": "MoveAhead", "R": "RotateRight", "L": "RotateLeft"}[a]

    def frontier(self, pose, visited, min_d=1.3):
        """Nearest observed free cell next to unknown space, away from where the agent has already been."""
        m = self.map
        free = m.known_free() & ~m.blocked(inflate=self.inflate)
        unknown = (m.free == 0) & (m.occ == 0) & ~m.forced_free
        nb = np.zeros_like(unknown)
        nb[1:, :] |= unknown[:-1, :]
        nb[:-1, :] |= unknown[1:, :]
        nb[:, 1:] |= unknown[:, :-1]
        nb[:, :-1] |= unknown[:, 1:]
        ii, jj = np.nonzero(free & nb)
        if ii.size == 0:
            return None
        xs, zs = m.origin[0] + (ii + 0.5) * m.res, m.origin[1] + (jj + 0.5) * m.res
        d = np.hypot(xs - pose["x"], zs - pose["z"])
        V = np.asarray(visited) if visited else np.zeros((0, 2))
        dv = np.min(np.hypot(xs[:, None] - V[None, :, 0], zs[:, None] - V[None, :, 1]), axis=1) if len(V) else d
        ok = (d > min_d) & (dv > 1.0)
        if not ok.any():
            return None
        k = int(np.argmin(np.where(ok, d, np.inf)))
        return np.array([xs[k], zs[k]])


class S2MRoboTHOR(S2MPlanner):
    name = "s2m_robothor"

    def __init__(self, model_name, stop_dist=0.9, stop_presence=0.5, scan_spread=40, **kw):
        core = HeadingCore()
        super().__init__(model_name, fov=core.fov, cam_h=0.88, core=core, use_exhaust=True, max_range=5.0, **kw)
        self.stop_dist, self.stop_presence, self.scan_spread = stop_dist, stop_presence, scan_spread

    def reset(self, instruction):
        super().reset(instruction)
        if isinstance(self.core, HeadingCore):
            self.core.reset_state()
        self.seen_bins, self.scan_origin, self.frontier_goal, self._last_action = set(), None, None, None
        self.events = []

    def _view_key(self, pose):
        # continuous positions: one magnified search per ~1 m cell and 60-degree heading sector
        return round(pose["x"]), round(pose["z"]), int(((pose["yaw"] % 360) + 30) // 60) % 6

    def _event(self, kind, h, **kw):
        self.events.append(dict(kind=kind, step=len(self.visited), xz=[round(float(v), 2) for v in h.xz],
                                score=round(h.score, 2), **kw))

    def _mark_seen(self, yaw):
        for b in range(12):
            if abs((b * ROT - yaw + 180) % 360 - 180) <= self.scan_spread:
                self.seen_bins.add(b)

    def act(self, obs):
        self.cam_h = obs["cam_h"]
        self.core.ensure(obs)
        pose = obs["pose"]
        p = np.array([pose["x"], pose["z"]])
        # perceive every step except plain forward moves while nothing is committed (view changes little)
        if not (self._last_action == "MoveAhead" and self._committed() is None and obs["step"] % 2 == 1):
            self.perceive(obs)
        self._last_pose = pose
        self.visited.append((pose["x"], pose["z"]))
        self._mark_seen(pose["yaw"])
        budget = max(1, 500 - obs["step"])
        a, info = self._decide(obs, pose, p, budget)
        self._last_action = a
        return a, info

    def _decide(self, obs, pose, p, budget):
        for _ in range(3):
            h = self._select(pose)
            if h is None:
                break
            d3 = math.hypot(float(np.linalg.norm(h.xz - p)), h.height - self.cam_h)
            if d3 <= self.stop_dist:
                b = self.core.bearing(pose, h.xz)
                if abs(b) > 20:
                    return ("RotateRight" if b > 0 else "RotateLeft"), self._info("aim", h)
                H, W = obs["depth"].shape
                px = world_to_pixel(h.xz[0], h.xz[1], h.height, pose, fov=self.fov, img_w=W, img_h=H, cam_h=self.cam_h)
                if px is None or px[1] > 0.8 * H:
                    if pose["horizon"] < 59:
                        return "LookDown", self._info("tilt", h)
                elif px[1] < 0.2 * H and pose["horizon"] > -29:
                    return "LookUp", self._info("tilt", h)
                pc = self.g.presence(obs["rgb"], self._query())
                if pc >= self.stop_presence:
                    self._event("stop", h, presence=round(pc, 3))
                    return "Stop", self._info("stop", h)
                self._event("reject", h, presence=round(pc, 3))
                h.exhausted = True  # within reach, centred, but not confirmed: a false hypothesis
                self.cur = None
                continue
            a = self.core.next_move(pose, h.xz, budget=budget)
            if a is not None:
                return a, self._info("approach", h)
            b = self.core.bearing(pose, h.xz)
            if abs(b) > 20:
                return ("RotateRight" if b > 0 else "RotateLeft"), self._info("push", h)
            if not self.map_move_failed(pose):
                return "MoveAhead", self._info("push", h)
            self._event("unreachable", h)
            h.exhausted = True  # unreachable from here
            self.cur = None
        if abs(pose["horizon"] - 30) > 5 and self._committed() is None:  # reported tilt is not exact
            return ("LookDown" if pose["horizon"] < 30 else "LookUp"), {"mode": "reset_tilt"}
        if self.scan_origin is None or np.linalg.norm(p - self.scan_origin) > 1.5:
            self.scan_origin, self.seen_bins = p.copy(), set()
            self._mark_seen(pose["yaw"])
        if len(self.seen_bins) < 12:
            return "RotateRight", {"mode": "scan", "seen": len(self.seen_bins)}
        live = self._live()
        h = max(live, key=lambda x: x.score) if live else None
        if h is not None and h.score >= self.inspect_th:
            a = self.core.next_move(pose, h.xz, budget=budget)
            if a is not None:
                return a, self._info("inspect", h)
        if self.frontier_goal is None or np.linalg.norm(self.frontier_goal - p) < 0.4:
            self.frontier_goal = self.core.frontier(pose, self.visited)
        if self.frontier_goal is not None:
            a = self.core.next_move(pose, self.frontier_goal, budget=budget)
            if a is not None:
                return a, {"mode": "explore", "goal": [round(float(v), 2) for v in self.frontier_goal]}
            self.frontier_goal = None
        return "RotateRight", {"mode": "stuck"}

    @property
    def stats(self):
        st = super().stats
        st["events"] = self.events[-20:]
        return st

    def map_move_failed(self, pose):
        fv, _ = fwd_right(pose["yaw"])
        return self.core.map.move_failed_before(np.array([pose["x"], pose["z"]]), fv)

    def observe(self, info, obs):
        if self._last_pose is not None:
            self.core.record_move(self._last_pose, info["action"], info["ok"])
        self.cam_h = obs["cam_h"]
        self.core.ensure(obs)
