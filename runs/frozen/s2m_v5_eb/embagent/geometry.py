"""Metric geometry for egocentric RGB-D navigation (conventions verified against AI2-THOR 5.0).

Frames
- world: x right / z forward at yaw 0, y up; yaw in degrees, clockwise (RotateRight adds +90).
- forward(yaw) = (sin yaw, cos yaw), right(yaw) = (cos yaw, -sin yaw) in (x, z).
- camera: planar depth (metres along the optical axis), square pinhole with vertical FOV `fov`;
  `horizon` > 0 means looking down.
The pose comes from the agent's odometry (the simulator's reported pose, optionally with injected noise).
"""
import heapq
import math

import numpy as np

CAM_HEIGHT = 1.576  # m above the floor for the default AI2-THOR agent
AGENT_RADIUS = 0.2


def fwd_right(yaw_deg):
    y = math.radians(yaw_deg)
    return np.array([math.sin(y), math.cos(y)]), np.array([math.cos(y), -math.sin(y)])


def focal(size, fov_deg):
    """Focal length in pixels for an image side of `size` px spanning `fov_deg` (AI2-THOR: vertical FOV)."""
    return (size / 2) / math.tan(math.radians(fov_deg) / 2)


def pixel_to_world(u, v, d, pose, fov=100.0, size=500, cam_h=CAM_HEIGHT, img_w=None, img_h=None):
    """Back-project pixel(s) (u right, v down) with planar depth d -> world (x, z, height above floor).
    `fov` is the vertical FOV over img_h (= size) rows; pixels are square, so the same focal applies to u."""
    img_w, img_h = img_w or size, img_h or size
    f = focal(img_h, fov)
    cu, cv = img_w / 2, img_h / 2
    xr = (np.asarray(u, float) - cu) / f * d         # right
    yd = (np.asarray(v, float) - cv) / f * d         # image down
    zc = np.asarray(d, float)                        # optical axis
    h = math.radians(pose.get("horizon", 0.0))
    fwd_d = zc * math.cos(h) - yd * math.sin(h)
    up = -(zc * math.sin(h) + yd * math.cos(h))
    fv, rv = fwd_right(pose["yaw"])
    x = pose["x"] + fwd_d * fv[0] + xr * rv[0]
    z = pose["z"] + fwd_d * fv[1] + xr * rv[1]
    return x, z, cam_h + up


def world_to_pixel(x, z, height, pose, fov=100.0, size=500, cam_h=CAM_HEIGHT, img_w=None, img_h=None):
    """Project a world point into the current image. Returns (u, v, depth) or None if behind the camera."""
    img_w, img_h = img_w or size, img_h or size
    fv, rv = fwd_right(pose["yaw"])
    rel = np.array([x - pose["x"], z - pose["z"]])
    fwd_d, xr, up = float(rel @ fv), float(rel @ rv), height - cam_h
    h = math.radians(pose.get("horizon", 0.0))
    zc = fwd_d * math.cos(h) - up * math.sin(h)
    yd = -fwd_d * math.sin(h) - up * math.cos(h)
    if zc <= 0.05:
        return None
    f = focal(img_h, fov)
    return img_w / 2 + f * xr / zc, img_h / 2 + f * yd / zc, zc


class OccupancyMap:
    """Top-down grid accumulated from depth. Cells: unknown / free floor / obstacle (with hit counts)."""

    def __init__(self, center, half_extent=8.0, res=0.05, fov=100.0, size=500, cam_h=CAM_HEIGHT):
        self.res, self.fov, self.size, self.cam_h = res, fov, size, cam_h
        self.origin = np.array([center[0] - half_extent, center[1] - half_extent])
        n = int(round(2 * half_extent / res))
        self.free = np.zeros((n, n), np.int16)
        self.occ = np.zeros((n, n), np.int16)
        self.forced_free = np.zeros((n, n), bool)
        self.n = n
        self._blocked = None
        self._blocked_r = None
        self.failed_moves = []  # (start_xz, unit_direction) of moves the simulator rejected

    def add_failed_move(self, start, direction):
        self.failed_moves.append((np.asarray(start, float), np.asarray(direction, float)))

    def move_failed_before(self, p, d, pos_tol=0.08):
        for s, u in self.failed_moves:
            if np.linalg.norm(s - p) < pos_tol and float(u @ d) > 0.9:
                return True
        return False

    def cell(self, x, z):
        i = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        j = np.floor((np.asarray(z) - self.origin[1]) / self.res).astype(int)
        return i, j

    def inside(self, i, j):
        return (i >= 0) & (j >= 0) & (i < self.n) & (j < self.n)

    def integrate(self, depth, pose, stride=4, max_range=5.0, floor_h=0.10, max_h=1.9):
        d = depth[::stride, ::stride]
        vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        ok = (d > 0.1) & (d < max_range)
        x, z, hgt = pixel_to_world(uu[ok], vv[ok], d[ok], pose, self.fov, self.size, cam_h=self.cam_h,
                                   img_w=depth.shape[1], img_h=depth.shape[0])
        i, j = self.cell(x, z)
        m = self.inside(i, j)
        i, j, hgt = i[m], j[m], hgt[m]
        floor = hgt < floor_h
        obst = (hgt >= floor_h) & (hgt <= max_h)
        np.add.at(self.free, (i[floor], j[floor]), 1)
        np.add.at(self.occ, (i[obst], j[obst]), 1)
        self._blocked = None

    def mark_free_disk(self, x, z, r=AGENT_RADIUS):
        """The agent's own footprint is free (it is standing there)."""
        i0, j0 = self.cell(x, z)
        k = int(math.ceil(r / self.res))
        ii, jj = np.mgrid[-k:k + 1, -k:k + 1]
        m = ii ** 2 + jj ** 2 <= k * k
        ii, jj = ii[m] + i0, jj[m] + j0
        ok = self.inside(ii, jj)
        self.forced_free[ii[ok], jj[ok]] = True
        self._blocked = None

    def mark_blocked_segment(self, a, b):
        """A move from a to b failed: treat the first part of the segment as an obstacle."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        for t in np.linspace(0.35, 1.0, 6):
            p = a + t * (b - a)
            i, j = self.cell(p[0], p[1])
            if self.inside(i, j) and not self.forced_free[i, j]:
                self.occ[i, j] += 5
        self._blocked = None

    def blocked(self, min_hits=2, inflate=AGENT_RADIUS):
        """Obstacle cells inflated by `inflate` metres (cached per radius until the map changes)."""
        if self._blocked is None or self._blocked_r != inflate:
            self._blocked_r = inflate
            self._blocked = None
        if self._blocked is None:
            occ = (self.occ >= min_hits) & ~self.forced_free
            k = int(math.ceil(inflate / self.res))
            out = occ.copy()
            if k > 0 and occ.any():
                ii, jj = np.nonzero(occ)
                offs = [(di, dj) for di in range(-k, k + 1) for dj in range(-k, k + 1) if di * di + dj * dj <= k * k]
                for di, dj in offs:
                    a, b = np.clip(ii + di, 0, self.n - 1), np.clip(jj + dj, 0, self.n - 1)
                    out[a, b] = True
            out &= ~self.forced_free
            self._blocked = out
        return self._blocked

    def known_free(self):
        return (self.free > 0) | self.forced_free


MOVE_DIRS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # EB move id -> (forward, right) sign


def plan_pose(occ_map, start, yaw, goal, step=0.25, max_cost=20, goal_tol=0.55, inflate=AGENT_RADIUS,
              unknown_cost=0.3):
    """A* over (lattice position, heading) with EB-Navigation actions, each costing one env step:
    0 fwd, 1 back, 2 right, 3 left (0.25 m), 4 rotate right 90, 5 rotate left 90.

    The camera only looks forward, so backward/sideways moves must stay inside observed free space;
    forward moves into unobserved cells are allowed at `unknown_cost` extra. Moves the simulator already
    rejected are never retried. Returns (actions, states, residual) where states[i] = (xz, yaw) after
    actions[i] and residual is the distance from the final state to the goal."""
    blocked = occ_map.blocked(inflate=inflate)
    known = occ_map.known_free()
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    yaw0 = int(round(yaw / 90.0) * 90) % 360
    dirs = {}
    for y in (0, 90, 180, 270):
        fv, rv = fwd_right(y)
        dirs[y] = {a: f * fv + r * rv for a, (f, r) in MOVE_DIRS.items()}

    def key(p, y):
        return round(p[0] * 100), round(p[1] * 100), y

    def h(p):
        return max(0.0, (abs(p[0] - goal[0]) + abs(p[1] - goal[1]) - goal_tol) / step)

    def move_cost(p, d, forward):
        if occ_map.failed_moves and occ_map.move_failed_before(p, d):
            return None
        extra = 0.0
        for t in (0.25, 0.5, 0.75, 1.0):
            r = p + t * step * d
            i, j = occ_map.cell(r[0], r[1])
            if not occ_map.inside(i, j) or blocked[i, j]:
                return None
            if not known[i, j]:
                if not forward:
                    return None
                extra += unknown_cost / 4
        return 1.0 + extra

    tie = 0
    openq = [(h(start), 0.0, tie, start, yaw0, None, None)]
    came = {}
    best = (float(np.linalg.norm(start - goal)), 0.0, key(start, yaw0))
    while openq:
        _, g, _, p, y, parent, act = heapq.heappop(openq)
        k = key(p, y)
        if k in came:
            continue
        came[k] = (p, y, parent, act)
        d = float(np.linalg.norm(p - goal))
        if d < best[0] - 1e-6 or (abs(d - best[0]) < 1e-6 and g < best[1]):
            best = (d, g, k)
        if d <= goal_tol:
            break
        if g + 1 > max_cost:
            continue
        for a, ny in ((4, (y + 90) % 360), (5, (y - 90) % 360)):
            if key(p, ny) not in came:
                tie += 1
                heapq.heappush(openq, (g + 1 + h(p), g + 1, tie, p, ny, k, a))
        for a, dv in dirs[y].items():
            q = p + step * dv
            if key(q, y) in came:
                continue
            c = move_cost(p, dv, a == 0)
            if c is None:
                continue
            tie += 1
            heapq.heappush(openq, (g + c + h(q), g + c, tie, q, y, k, a))
    acts, states, k = [], [], best[2]
    while k is not None:
        p, y, parent, a = came[k]
        if a is not None:
            acts.append(a)
            states.append((p, y))
        k = parent
    return acts[::-1], states[::-1], best[0]


def move_is_valid(occ_map, p, yaw, action, step=0.25, inflate=AGENT_RADIUS):
    """Re-check one planned move against the current map (same rules as plan_pose)."""
    if action not in MOVE_DIRS:
        return True
    fv, rv = fwd_right(int(round(yaw / 90.0) * 90) % 360)
    f, r = MOVE_DIRS[action]
    d = f * fv + r * rv
    p = np.asarray(p, float)
    if occ_map.failed_moves and occ_map.move_failed_before(p, d):
        return False
    blocked, known = occ_map.blocked(inflate=inflate), occ_map.known_free()
    for t in (0.25, 0.5, 0.75, 1.0):
        q = p + t * step * d
        i, j = occ_map.cell(q[0], q[1])
        if not occ_map.inside(i, j) or blocked[i, j] or (action != 0 and not known[i, j]):
            return False
    return True


def plan_lattice(occ_map, start, goal, yaw, step=0.25, max_moves=24, unknown_cost=0.6, goal_tol=0.55,
                 inflate=AGENT_RADIUS):
    """A* over a 0.25 m lattice of translations (no rotation needed: moves are along the 4 world axes when
    yaw is a multiple of 90). Returns (path_points, actions) where actions are EB action ids
    (0 fwd, 1 back, 2 right, 3 left) expressed relative to the current yaw.

    The goal is the reachable lattice node closest to `goal` (stop early if within goal_tol)."""
    blocked = occ_map.blocked(inflate=inflate)
    known = occ_map.known_free()
    fv, rv = fwd_right(yaw)
    dirs = {0: fv, 1: -fv, 2: rv, 3: -rv}

    def edge_cost(p, q):
        if occ_map.failed_moves and occ_map.move_failed_before(p, (q - p) / step):
            return None
        cost = 1.0
        for t in (0.25, 0.5, 0.75, 1.0):
            r = p + t * (q - p)
            i, j = occ_map.cell(r[0], r[1])
            if not occ_map.inside(i, j) or blocked[i, j]:
                return None
            if not known[i, j]:
                cost += unknown_cost / 4
        return cost

    start = np.asarray(start, float)
    goal = np.asarray(goal, float)

    def key(p):
        return round(p[0] * 100), round(p[1] * 100)

    def h(p):  # admissible: Manhattan distance minus the tolerance, in moves
        return max(0.0, (abs(p[0] - goal[0]) + abs(p[1] - goal[1]) - goal_tol) / step)

    tie = 0
    openq = [(h(start), 0.0, tie, 0, start, None, None)]
    came = {}  # key -> (point, parent_key, action, depth), fixed when the node is popped
    best = (float(np.linalg.norm(start - goal)), 0, key(start))
    while openq:
        _, g, _, depth, p, parent, act = heapq.heappop(openq)
        k = key(p)
        if k in came:
            continue
        came[k] = (p, parent, act, depth)
        d = float(np.linalg.norm(p - goal))
        if d < best[0] - 1e-6 or (abs(d - best[0]) < 1e-6 and depth < best[1]):
            best = (d, depth, k)
        if d <= goal_tol:
            break
        if depth >= max_moves:
            continue
        for a, dv in dirs.items():
            q = p + step * dv
            if key(q) in came:
                continue
            c = edge_cost(p, q)
            if c is None:
                continue
            tie += 1
            heapq.heappush(openq, (g + c + h(q), g + c, tie, depth + 1, q, k, a))
    path, acts, k = [], [], best[2]
    while k is not None:
        p, parent, a, _ = came[k]
        path.append(p)
        if a is not None:
            acts.append(a)
        k = parent
    return path[::-1], acts[::-1], best[0]


def plan_heading(occ_map, start, yaw, goal, step=0.25, rot=30, max_cost=60, goal_tol=0.55, inflate=AGENT_RADIUS,
                 unknown_cost=0.3, key_res=0.1):
    """A* for forward-only robots (e.g. RoboTHOR LoCoBot): actions 'F' (move `step` m ahead), 'R'/'L' (rotate
    by `rot` degrees), each costing one step. Headings are kept on the multiples of `rot` nearest the start yaw;
    positions are keyed on a `key_res` grid because continuous motion does not stay on a lattice.
    Forward moves may enter unobserved cells at `unknown_cost` extra (the camera sees ahead).
    Returns (actions, states, residual) as in plan_pose."""
    blocked = occ_map.blocked(inflate=inflate)
    known = occ_map.known_free()
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    n_head = int(round(360 / rot))
    h0 = yaw % 360

    def key(p, k):
        return round(p[0] / key_res), round(p[1] / key_res), k % n_head

    def yaw_of(k):
        return (h0 + rot * k) % 360

    def h(p):  # admissible: straight-line distance minus tolerance, in steps
        return max(0.0, (float(np.linalg.norm(p - goal)) - goal_tol) / step)

    def move_cost(p, d):
        if occ_map.failed_moves and occ_map.move_failed_before(p, d):
            return None
        extra = 0.0
        for t in (0.25, 0.5, 0.75, 1.0):
            r = p + t * step * d
            i, j = occ_map.cell(r[0], r[1])
            if not occ_map.inside(i, j) or blocked[i, j]:
                return None
            if not known[i, j]:
                extra += unknown_cost / 4
        return 1.0 + extra

    tie = 0
    openq = [(h(start), 0.0, tie, start, 0, None, None)]
    came = {}
    best = (float(np.linalg.norm(start - goal)), 0.0, key(start, 0))
    while openq:
        _, g, _, p, k, parent, act = heapq.heappop(openq)
        kk = key(p, k)
        if kk in came:
            continue
        came[kk] = (p, k, parent, act)
        d = float(np.linalg.norm(p - goal))
        if d < best[0] - 1e-6 or (abs(d - best[0]) < 1e-6 and g < best[1]):
            best = (d, g, kk)
        if d <= goal_tol:
            break
        if g + 1 > max_cost:
            continue
        for a, nk in (("R", k + 1), ("L", k - 1)):
            if key(p, nk) not in came:
                tie += 1
                heapq.heappush(openq, (g + 1 + h(p), g + 1, tie, p, nk, kk, a))
        fv, _ = fwd_right(yaw_of(k))
        q = p + step * fv
        if key(q, k) not in came:
            c = move_cost(p, fv)
            if c is not None:
                tie += 1
                heapq.heappush(openq, (g + c + h(q), g + c, tie, q, k, kk, "F"))
    acts, states, kk = [], [], best[2]
    while kk is not None:
        p, k, parent, a = came[kk]
        if a is not None:
            acts.append(a)
            states.append((p, yaw_of(k)))
        kk = parent
    return acts[::-1], states[::-1], best[0]
