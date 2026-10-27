"""S2M: semantic-to-metric navigation agent for small local VLMs.

The VLM only answers semantic questions about the current image (is the target visible, where is it in the
image); every metric decision (3D target estimate, map, path, search coverage) is made by geometry from
RGB-D + odometry. Target hypotheses are fused across *distinct* views in 3D with positive and negative
evidence; the agent commits to a hypothesis when the evidence is sufficient, keeps it unless the evidence
collapses (hysteresis), and abandons it if reaching it does not finish the task. When nothing is found in the
full frame, a magnified tiled search with crop verification looks for small, distant targets.
"""
import concurrent.futures as cf

import numpy as np

from embagent.geometry import world_to_pixel
from embagent.grounding import Grounder, _iou, lift_box
from embagent.planners import Planner
from embagent.planners.geo import GeoCore, snap_yaw


class Hypothesis:
    def __init__(self, xz, height, w, view):
        self.xz, self.height = np.asarray(xz, float), height
        self.score, self.hits, self.misses = w, 1, 0
        self.views = [view]
        self.exhausted = False
        self.peak = w  # highest evidence ever reached (logging / RoboTHOR revisit)

    def merge(self, xz, height, w, view, same_view):
        n = self.hits
        self.xz = (self.xz * n + np.asarray(xz)) / (n + 1)
        self.height = (self.height * n + height) / (n + 1)
        self.hits += 1
        # repeated looks from (almost) the same pose add little new information
        self.score += 0.25 * w if same_view else w
        self.peak = max(self.peak, self.score)
        if not same_view:
            self.views.append(view)


class S2MPlanner(Planner):
    name = "s2m"

    def __init__(self, model_name, merge_radius=0.5, commit=0.8, keep=0.4, inspect=0.35, p_present=0.5,
                 zoom_accept=0.5, neg_weight=0.4, tile_grid=3, max_tile_boxes=4, max_range=6.0,
                 use_negative=True, use_scan=True, use_zoom=True, use_tiles=True, use_exhaust=True,
                 use_fusion=True, nearest_first=True, tie_margin=0.3, fov=100.0, cam_h=None, core=None, use_contrast=False, **kw):
        self.g = Grounder(model_name)
        self.core = core or GeoCore()
        self.fov, self.cam_h = fov, cam_h
        self.merge_radius, self.commit, self.keep, self.inspect_th = merge_radius, commit, keep, inspect
        self.p_present, self.zoom_accept, self.neg_weight = p_present, zoom_accept, neg_weight
        self.tile_grid, self.max_tile_boxes, self.max_range = tile_grid, max_tile_boxes, max_range
        self.use_negative, self.use_scan, self.use_zoom, self.use_tiles = use_negative, use_scan, use_zoom, use_tiles
        self.use_exhaust, self.use_fusion, self.nearest_first = use_exhaust, use_fusion, nearest_first
        self.tie_margin = tie_margin
        self.use_contrast = use_contrast
        self.pool = cf.ThreadPoolExecutor(2)
        self.tpool = cf.ThreadPoolExecutor(9)
        self.reset("")

    # ---- episode ------------------------------------------------------------------------------
    def reset(self, instruction):
        self.instruction = instruction
        self.spec = None
        self.hyps, self.cur, self.seen_headings, self.log = [], None, set(), []
        self.visited, self.explore_goal, self._tiled_views = [], None, set()
        self.core.map = None
        self._last_pose = None
        self.g.calls = 0

    def _query(self):
        if self.spec is None:
            self.spec = self.g.target_spec(self.instruction)
        return self.spec["query"]

    # ---- perception ---------------------------------------------------------------------------
    def perceive(self, obs):
        rgb, depth, pose = obs["rgb"], obs["depth"], obs["pose"]
        q = self._query()
        f_vis = self.pool.submit(self.g.presence, rgb, q)
        f_box = self.pool.submit(self.g.ground, rgb, q)
        p_vis, (box, _) = f_vis.result(), f_box.result()
        dets, p_zoom, tiled = [], None, 0
        named = self.spec.get("named", True)
        cands = []
        if self.use_contrast and (box is not None or p_vis >= 0.2):
            cands = self.g.ground_all(rgb, q)
            if box is not None and all(_iou(box, b) < 0.5 for b in cands):
                cands = (cands + [box])[:6]
        if len(cands) >= 2:
            # relative verification: which of the visible candidates is the target (Set-of-Mark choice)
            probs = self.g.choose(rgb, cands, q)
            p_zoom = round(float(probs[0]), 3)  # logged as P(none)
            for b, pr in zip(cands, probs[1:]):
                d = self._lift(b, depth, pose, float(pr) * max(p_vis, 0.5))
                if d is not None:
                    dets.append(d)
            box = None
        if box is not None:
            if p_vis >= self.p_present and (named or not self.use_zoom):
                w = p_vis
            elif self.use_zoom:  # no image-level agreement, or an appearance-only target: check attributes up close
                p_zoom = self.g.zoom_presence(rgb, box, q)
                w = p_zoom if p_zoom >= self.zoom_accept else 0.2 * p_zoom
            else:
                w = 0.25 * p_vis
            d = self._lift(box, depth, pose, w)
            if d is not None:
                dets.append(d)
        if self.use_tiles and not any(d[2] >= self.zoom_accept for d in dets) and self._committed() is None:
            tiled_dets = self._tiled_search(rgb, depth, pose, q)
            tiled = len(tiled_dets)
            dets += tiled_dets
        for d in dets:
            self._add(d, pose)
        if self.use_negative:
            self._negative_evidence(depth, pose, dets)
        self.seen_headings.add(snap_yaw(pose["yaw"]))
        self.log.append({"step": obs["step"], "p_vis": round(p_vis, 3),
                         "p_zoom": None if p_zoom is None else round(p_zoom, 3), "tiled": tiled,
                         "box": None if box is None else [int(v) for v in box],
                         "dets": [[round(float(v), 2) for v in d[0]] + [round(d[2], 2)] for d in dets],
                         "hyps": [(round(float(h.xz[0]), 2), round(float(h.xz[1]), 2), round(h.score, 2), h.exhausted)
                                  for h in self.hyps]})

    def _lift(self, box, depth, pose, w):
        lifted = lift_box(box, depth, pose, fov=self.fov, cam_h=self.cam_h)
        if lifted is None or lifted[3] > self.max_range or w <= 0.02:
            return None
        return np.array(lifted[:2]), lifted[2], w

    def _tiled_search(self, rgb, depth, pose, q):
        """Magnified tiles for small/distant targets; every tile box must pass a crop presence check."""
        key = self._view_key(pose)
        if key in self._tiled_views:
            return []
        self._tiled_views.add(key)
        boxes = [b for b in self.tpool.map(lambda t: self.g.ground_tile(t, q), self.g.tiles(rgb, self.tile_grid))
                 if b is not None]
        uniq = []
        for b in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if all(abs(c[0] - (u[0] + u[2]) / 2) + abs(c[1] - (u[1] + u[3]) / 2) > 25 for u in uniq):
                uniq.append(b)
        uniq = uniq[:self.max_tile_boxes]
        scores = list(self.tpool.map(lambda b: self.g.zoom_presence(rgb, b, q), uniq))
        out = []
        for b, pz in zip(uniq, scores):
            if pz >= self.zoom_accept:
                d = self._lift(b, depth, pose, pz)
                if d is not None:
                    out.append(d)
        return out

    def _view_key(self, pose):
        """Views that count as 'the same' for the once-per-view magnified search."""
        return round(pose["x"], 1), round(pose["z"], 1), snap_yaw(pose["yaw"])

    def _add(self, det, pose):
        xz, hgt, w = det
        view = (pose["x"], pose["z"], snap_yaw(pose["yaw"]))
        if self.use_fusion:
            for h in self.hyps:
                if np.linalg.norm(h.xz - xz) < self.merge_radius:
                    same = any(abs(v[0] - view[0]) + abs(v[1] - view[1]) < 0.2 and v[2] == view[2] for v in h.views)
                    h.merge(xz, hgt, w, view, same)
                    return
        else:  # ablation: trust the latest single-view detection only
            self.hyps = [h for h in self.hyps if h.exhausted]
            self.cur = None
        self.hyps.append(Hypothesis(xz, hgt, w, view))

    def _negative_evidence(self, depth, pose, dets):
        """A hypothesis that projects into the image, unoccluded, but is not detected loses evidence."""
        H, W = depth.shape
        for h in self.hyps:
            if any(np.linalg.norm(h.xz - d[0]) < self.merge_radius for d in dets):
                continue
            kw = {} if self.cam_h is None else {"cam_h": self.cam_h}
            px = world_to_pixel(h.xz[0], h.xz[1], h.height, pose, fov=self.fov, img_w=W, img_h=H, **kw)
            if px is None:
                continue
            u, v, d = px
            if not (20 <= u < W - 20 and 20 <= v < H - 20) or d > self.max_range:
                continue
            if depth[int(v), int(u)] + 0.3 < d:  # occluded by something nearer
                continue
            h.score -= self.neg_weight
            h.misses += 1
        self.hyps = [h for h in self.hyps if h.score > -1.0 or h.exhausted]
        if self.cur is not None and self.cur not in self.hyps:
            self.cur = None

    def _live(self):
        return [h for h in self.hyps if not h.exhausted]

    def _committed(self):
        return self.cur if (self.cur is not None and not self.cur.exhausted and self.cur.score >= self.keep) else \
            next((h for h in self._live() if h.score >= self.commit), None)

    def _select(self, pose):
        """Keep the current target unless its evidence collapsed; otherwise commit to the nearest strong one."""
        if self.cur is not None and not self.cur.exhausted and self.cur.score >= self.keep and self.cur in self.hyps:
            return self.cur
        cands = [h for h in self._live() if h.score >= self.commit]
        if not cands:
            self.cur = None
            return None
        p = np.array([pose["x"], pose["z"]])
        if self.nearest_first:
            # optimal sequential search: visit the candidate with the largest P(target) / cost-to-check.
            # P from the accumulated evidence, cost = steps to reach it (+1 to check); this reduces to
            # nearest-first for equal evidence and to strongest-first for equal distance
            def index(h):
                prob = 1.0 - np.exp(-max(h.score, 0.0))
                return prob / (np.linalg.norm(h.xz - p) / 0.25 + 1.0)
            self.cur = max(cands, key=index)
        else:
            self.cur = max(cands, key=lambda h: h.score)
        return self.cur

    # ---- policy -------------------------------------------------------------------------------
    def act(self, obs):
        self.core.ensure(obs)
        self.perceive(obs)
        pose = obs["pose"]
        self._last_pose = pose
        self.visited.append((pose["x"], pose["z"]))
        budget = max(1, 20 - obs["step"])
        for _ in range(3):  # at most a few hypothesis switches per step
            h = self._select(pose)
            if h is None:
                break
            a = self.core.next_move(pose, h.xz, budget=budget)
            if a is not None:
                return [a], self._info("approach", h)
            resid = self.core.resid if self.core.resid is not None else 9.0
            if self.use_exhaust and resid <= 0.9:
                # we stand next to it and the episode is still running: not the target
                h.exhausted = True
                self.cur = None
                self.core.plan = []
                continue
            return [self.core.push_action(pose, h.xz)], self._info("push", h)
        if self.use_scan:  # finish the cheap look-around before chasing weak evidence
            yaw = snap_yaw(pose["yaw"])
            unseen = [d for d in (90, 270, 180) if (yaw + d) % 360 not in self.seen_headings]
            if unseen:
                return [4 if unseen[0] != 270 else 5], {"mode": "scan", "unseen": len(unseen)}
        live = self._live()
        h = max(live, key=lambda x: x.score) if live else None
        if h is not None and h.score >= self.inspect_th:  # weak hypothesis: walk towards it to look closer
            a = self.core.next_move(pose, h.xz, budget=budget)
            if a is not None:
                return [a], self._info("inspect", h)
        return [self._explore(pose, budget)], {"mode": "explore",
                                               "goal": None if self.explore_goal is None else
                                               [round(float(v), 2) for v in self.explore_goal]}

    def _info(self, mode, h):
        return {"mode": mode, "target": [round(float(v), 2) for v in h.xz], "score": round(h.score, 2)}

    def _explore(self, pose, budget):
        """Go to a persistent vantage point far from everywhere the agent has stood, then look around again."""
        p = np.array([pose["x"], pose["z"]])
        if self.explore_goal is not None and np.linalg.norm(self.explore_goal - p) < 0.35:
            self.explore_goal = None
            if budget >= 10:
                self.seen_headings = {snap_yaw(pose["yaw"])}
        if self.explore_goal is None:
            self.explore_goal = self._pick_vantage(p)
            if self.explore_goal is None:
                return 4
        a = self.core.next_move(pose, self.explore_goal, budget=budget)
        if a is None:
            self.explore_goal = None
            return 4
        return a

    def _pick_vantage(self, p):
        m = self.core.map
        ok = m.known_free() & ~m.blocked()
        ii, jj = np.nonzero(ok[::2, ::2])
        if ii.size == 0:
            return None
        xs = m.origin[0] + (2 * ii + 1) * m.res
        zs = m.origin[1] + (2 * jj + 1) * m.res
        d = np.hypot(xs - p[0], zs - p[1])
        sel = (d > 0.75) & (d < 3.0)
        if not sel.any():
            return None
        xs, zs = xs[sel], zs[sel]
        V = np.asarray(self.visited)
        dv = np.min(np.hypot(xs[:, None] - V[None, :, 0], zs[:, None] - V[None, :, 1]), axis=1)
        k = int(np.argmax(dv))
        return np.array([xs[k], zs[k]])

    def observe(self, info, obs):
        if self._last_pose is not None:
            self.core.record_move(self._last_pose, info["action_id"], bool(info["last_action_success"]))
        self.core.ensure(obs)

    @property
    def stats(self):
        return {"vlm_calls": self.g.calls, "spec": self.spec, "perception_log": self.log[-25:]}
