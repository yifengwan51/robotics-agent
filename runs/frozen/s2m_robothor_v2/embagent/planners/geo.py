"""Geometry core for metric navigation from RGB-D + odometry, and an oracle-target diagnostic planner.

GeoCore keeps a depth-built occupancy map and turns a metric target estimate into EB-Navigation actions
(0 fwd, 1 back, 2 right, 3 left, 4 rotate right 90, 5 rotate left 90). It never reads simulator state
beyond what a robot would have: RGB-D frames, its odometry pose and whether the last move succeeded.

Plans are committed: the agent follows the current lattice path and replans only when a move fails, it
drifts off the path (the simulator snaps positions to a 0.1 m grid), or the target estimate moves.
"""
import math

import numpy as np

from embagent.geometry import OccupancyMap, fwd_right, move_is_valid, plan_pose
from embagent.planners import Planner

MOVE_DELTA = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # (forward, right) units of 0.25 m


def snap_yaw(yaw):
    return (round(yaw / 90.0) * 90) % 360


def move_action(pose, direction):
    """EB move action (0-3) whose world direction best matches `direction` at the current yaw."""
    fv, rv = fwd_right(snap_yaw(pose["yaw"]))
    cand = {0: fv, 1: -fv, 2: rv, 3: -rv}
    return max(cand, key=lambda a: float(cand[a] @ direction))


class GeoCore:
    def __init__(self, goal_tol=0.55, unknown_cost=0.6, step=0.25, inflate=0.2, tight_inflate=0.1,
                 replan_drift=0.12, target_shift=0.3, fov=100.0, size=500):
        self.goal_tol, self.unknown_cost, self.step = goal_tol, unknown_cost, step
        self.inflate, self.tight_inflate = inflate, tight_inflate
        self.replan_drift, self.target_shift = replan_drift, target_shift
        self.fov, self.size = fov, size
        self.map = None
        self._integrated_step = -1
        self.plan, self.plan_target, self.resid, self._expect, self.plan_inflate = [], None, None, None, inflate

    def reset(self, pose):
        self.map = OccupancyMap((pose["x"], pose["z"]), fov=self.fov, size=self.size)
        self._integrated_step = -1
        self.plan, self.plan_target, self.resid, self._expect = [], None, None, None

    def ensure(self, obs):
        """Integrate this step's observation once (called from both act and observe)."""
        if self.map is None:
            self.reset(obs["pose"])
        if obs["step"] == self._integrated_step:
            return
        pose = obs["pose"]
        self.map.mark_free_disk(pose["x"], pose["z"])
        if obs.get("depth") is not None:
            self.map.integrate(obs["depth"], pose)
        self._integrated_step = obs["step"]

    def record_move(self, pose_before, action, ok):
        if action in MOVE_DELTA and not ok:
            f, r = MOVE_DELTA[action]
            fv, rv = fwd_right(snap_yaw(pose_before["yaw"]))
            a = np.array([pose_before["x"], pose_before["z"]])
            d = f * fv + r * rv
            self.map.mark_blocked_segment(a, a + self.step * d)
            self.map.add_failed_move(a, d)  # survives forced-free footprint cells
            self.plan = []  # replan around the new obstacle

    def bearing(self, pose, target):
        """Signed angle (deg) from the camera forward axis to the target; positive = to the right."""
        fv, rv = fwd_right(pose["yaw"])
        rel = np.array([target[0] - pose["x"], target[1] - pose["z"]])
        return math.degrees(math.atan2(rel @ rv, rel @ fv))

    def turn_towards(self, pose, target, tol=45):
        b = self.bearing(pose, target)
        if abs(b) <= tol:
            return None
        return 4 if b > 0 else 5

    def _replan(self, pos, target, yaw, budget):
        best = None
        for infl in (self.inflate, self.tight_inflate):
            acts, states, resid = plan_pose(self.map, pos, yaw, target, step=self.step, max_cost=budget,
                                            goal_tol=self.goal_tol, inflate=infl, unknown_cost=self.unknown_cost)
            if best is None or resid < best[2] - 0.05:
                best = (acts, states, resid, infl)
            if resid <= 0.85:  # comfortably inside the 1 m success radius
                break
        self.plan = list(zip(best[0], best[1]))
        self.plan_inflate = best[3]
        self.plan_target, self.resid = np.asarray(target, float), best[2]

    def next_move(self, pose, target, budget=20):
        """Next EB action towards the metric target (moves and rotations), or None if nothing is left.

        Follows a committed (position, heading) plan; replans if the target estimate moved, the last action
        did not land where expected (snapping, collisions), or the next move became invalid on the map."""
        pos = np.array([pose["x"], pose["z"]])
        yaw = snap_yaw(pose["yaw"])
        target = np.asarray(target, float)
        stale = (not self.plan or self.plan_target is None
                 or np.linalg.norm(self.plan_target - target) > self.target_shift)
        if not stale and self._expect is not None:
            exp_p, exp_y = self._expect
            if np.linalg.norm(exp_p - pos) > self.replan_drift or exp_y != yaw:
                stale = True
        if not stale and not move_is_valid(self.map, pos, yaw, self.plan[0][0], self.step, self.plan_inflate):
            stale = True
        if stale:
            self._replan(pos, target, yaw, budget)
        if not self.plan:
            self._expect = None
            return None
        action, (p_next, y_next) = self.plan.pop(0)
        self._expect = (np.asarray(p_next), y_next)
        return action

    def push_action(self, pose, target):
        """Plan exhausted but not yet successful: face the target, then try untried moves towards it."""
        turn = self.turn_towards(pose, target)
        if turn is not None:
            return turn
        pos = np.array([pose["x"], pose["z"]])
        fv, rv = fwd_right(snap_yaw(pose["yaw"]))
        to_t = np.asarray(target, float) - pos
        cands = sorted({0: fv, 2: rv, 3: -rv}.items(), key=lambda kv: -float(kv[1] @ to_t))
        for a, d in cands:
            if not self.map.move_failed_before(pos, d):
                return a
        return 4  # everything towards the target failed: rotate to look for another way

    def seen_free(self, a, b):
        """True if every sample on segment a->b (beyond the agent's own footprint) was observed as floor."""
        known = self.map.known_free()
        for t in (0.6, 0.8, 1.0, 1.2):
            p = a + t * (b - a)
            i, j = self.map.cell(p[0], p[1])
            if not self.map.inside(i, j) or not known[i, j]:
                return False
        return True


class OracleGeoPlanner(Planner):
    """DIAGNOSTIC ONLY: navigates to the ground-truth target position with the geometry core.
    Measures the success-rate ceiling of the metric 'how' sub-problem when 'what/where' is solved."""
    name = "oracle_geo"
    needs_oracle = True

    def __init__(self, face_target=False, **kw):
        self.core = GeoCore()
        self.face_target = face_target
        self._last_pose = None

    def reset(self, instruction):
        self.core.map = None
        self._last_pose = None

    def act(self, obs):
        self.core.ensure(obs)
        self._last_pose = obs["pose"]
        pose, target = obs["pose"], obs["oracle_target"]
        if self.face_target:
            turn = self.core.turn_towards(pose, target)
            if turn is not None:
                return [turn], {"why": "face target"}
        a = self.core.next_move(pose, target, budget=max(1, 20 - obs["step"]))
        if a is not None:
            return [a], {"why": f"follow plan ({len(self.core.plan)} left, residual {self.core.resid:.2f} m)"}
        return [self.core.push_action(pose, target)], {"why": f"push (residual {self.core.resid})"}

    def observe(self, info, obs):
        if self._last_pose is not None:  # pose before this action
            self.core.record_move(self._last_pose, info["action_id"], bool(info["last_action_success"]))
        self.core.ensure(obs)
        self._last_pose = obs["pose"]
