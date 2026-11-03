"""One EB-Navigation episode in AI2-THOR, with smooth motion for the viewer.

The agent and its planner see exactly what the benchmark gives them: the agent camera after each discrete action.
A separate display camera (an AI2-THOR third-party camera with the same position, orientation, and field of view)
is moved through the in-between poses of every action, so a 0.25 m step or a 90 degree turn is shown as motion
instead of a jump. Display frames are never given to a planner.
"""
import math
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EB_DATA = os.path.join(ROOT, "third_party", "EmbodiedBench", "embodiedbench", "envs", "eb_navigation", "datasets")
SUBSETS = ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"]
MOVES = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}            # action id -> (forward, right)
DURATION = {0: .6, 1: .6, 2: .6, 3: .6, 4: .9, 5: .9, 6: .45, 7: .45}  # seconds of motion per action


def ease(t, ramp=.2):
    """Trapezoidal velocity profile: accelerate for `ramp`, cruise, decelerate; peak speed 1.25x the mean."""
    v = 1.0 / (1 - ramp)
    if t < ramp:
        return v * t * t / (2 * ramp)
    if t > 1 - ramp:
        return 1 - v * (1 - t) ** 2 / (2 * ramp)
    return v * (t - ramp / 2)


def pose_of(ev):
    a = ev.metadata["agent"]
    return {"x": a["position"]["x"], "z": a["position"]["z"], "yaw": a["rotation"]["y"], "horizon": a["cameraHorizon"]}


def cam_of(ev):
    p, a = ev.metadata["cameraPosition"], ev.metadata["agent"]
    return {"x": p["x"], "y": p["y"], "z": p["z"], "yaw": a["rotation"]["y"], "horizon": a["cameraHorizon"]}


def dataset_path(split, subset):
    return os.path.join(ROOT, "data", "eb_nav_dev", f"{subset}.json") if split == "dev" else \
        os.path.join(EB_DATA, f"{subset}.json")


class AgentSim:
    def __init__(self, resolution=500):
        import sys
        sys.path.insert(0, ROOT)
        from embagent.eb import shim
        shim.install()
        from embodiedbench.envs.eb_navigation import EBNavEnv
        self.env = EBNavEnv.EBNavigationEnv(eval_set="base", exp_name="demo", resolution=resolution, selected_indexes=[0])
        self.env.log_path = os.path.join(tempfile.gettempdir(), "s2m_demo_eb_logs")
        self.c, self.skills = self.env.env, EBNavEnv.DISCRETE_SKILLSET
        self.fov, self.budget = self.env.config["fieldOfView"], self.env._max_episode_steps
        self.full = self.cam = self.disp = None

    def load(self, split, subset):
        self.env.data_path = dataset_path(split, subset)
        self.full = self.env._load_dataset(subset)
        return self.full

    def reset(self, idx):
        """Reset episode idx; returns (observation, reachable floor, target xz, instruction)."""
        e = self.env
        e.dataset, e.selected_indexes = [self.full[idx]], [idx]
        e.number_of_episodes, e._current_episode_num = 1, 0
        e.reset()
        ev = self.c.last_event
        obs = {"rgb": ev.frame.copy(), "depth": ev.depth_frame.copy(), "pose": pose_of(ev)}
        self.cam = cam_of(ev)
        floor = [(q["x"], q["z"]) for q in self.c.step(action="GetReachablePositions").metadata["actionReturn"]]
        c = self.cam
        ev = self.c.step(action="AddThirdPartyCamera", position={"x": c["x"], "y": c["y"], "z": c["z"]},
                         rotation={"x": c["horizon"], "y": c["yaw"], "z": 0}, fieldOfView=self.fov)
        self.disp = ev.third_party_camera_frames[0][:, :, :3]
        tp = self.full[idx]["target_position"]
        return obs, floor, (tp["x"], tp["z"]), e.episode_language_instruction

    def step(self, action, first=False):
        """Execute one benchmark action; returns (observation, info, done, camera before, camera after)."""
        c0 = self.cam
        _, _, done, info = self.env.step(action, {}, 1 if first else 0)
        ev = self.c.last_event
        obs = {"rgb": ev.frame.copy(), "depth": ev.depth_frame.copy(), "pose": pose_of(ev)}
        self.cam = cam_of(ev)
        return obs, info, done, c0, self.cam

    def show(self, cam):
        """Render the display camera at an arbitrary pose (does not move the agent)."""
        ev = self.c.step(action="UpdateThirdPartyCamera", thirdPartyCameraId=0,
                         position={"x": cam["x"], "y": cam["y"], "z": cam["z"]},
                         rotation={"x": cam["horizon"], "y": cam["yaw"], "z": 0}, fieldOfView=self.fov)
        self.disp = ev.third_party_camera_frames[0][:, :, :3]
        return self.disp

    @staticmethod
    def motion(c0, c1, action, ok, fps):
        """Display-camera poses from c0 to c1 (eased); a rejected move bumps 6 cm toward the obstacle and back."""
        n = max(2, round(DURATION.get(action, .4) * fps))
        if not ok and action in MOVES:
            f, r = MOVES[action]
            y = math.radians(c0["yaw"])
            dx, dz = .06 * (f * math.sin(y) + r * math.cos(y)), .06 * (f * math.cos(y) - r * math.sin(y))
            return [dict(c0, x=c0["x"] + dx * math.sin(math.pi * k / n), z=c0["z"] + dz * math.sin(math.pi * k / n))
                    for k in range(1, n + 1)]
        dyaw = (c1["yaw"] - c0["yaw"] + 180) % 360 - 180
        out = []
        for k in range(1, n + 1):
            s = ease(k / n)
            out.append({"x": c0["x"] + s * (c1["x"] - c0["x"]), "y": c0["y"] + s * (c1["y"] - c0["y"]),
                        "z": c0["z"] + s * (c1["z"] - c0["z"]), "yaw": (c0["yaw"] + s * dyaw) % 360,
                        "horizon": c0["horizon"] + s * (c1["horizon"] - c0["horizon"])})
        return out

    def close(self):
        try:
            self.c.stop()
        except Exception:
            pass


def episode(sim, src, panel, obs, instruction, fps, pause=0.6, stop=None):
    """Drive one episode. Yields ("think", panel, display_rgb, pose) before every model decision, ("frame", img, seconds)
    for every frame to show, ("step", info) after every action, and finally ("done", result)."""
    nan = float("nan")
    T, budget = panel.T, sim.budget
    traj, status, info, step, done, aborted = [(obs["pose"]["x"], obs["pose"]["z"])], [], {}, 0, False, False
    src.instruction = instruction
    src.reset(obs)
    vlm, dec = src.text(T, None, None, None, 0, None)
    hyps, goal = (), None
    panel.set_key(0, status, traj, vlm, dec, occ_map=src.occ_map, cur=0)
    yield ("frame", panel.frame(sim.disp, obs["pose"]), 2 * pause)
    while not done and not src.exhausted(step):
        if stop and stop():
            aborted = True
            break
        yield ("think", panel, sim.disp, obs["pose"])
        actions, reasoning, entry = src.decide(dict(obs, step=step))
        hyps = [tuple(h) for h in entry["hyps"]] if entry else []
        r = reasoning or {}
        goal = tuple(r.get("target") or r.get("goal") or ()) or None if src.agent == "s2m" else None
        vlm, dec = src.text(T, reasoning, entry, actions, 0, None)
        panel.set_key(step, status, traj, vlm, dec, hyps, goal, src.occ_map, cur=step)
        yield ("frame", panel.frame(sim.disp, obs["pose"], box=entry.get("box") if entry else None), pause)
        for j, a in enumerate(actions[:max(1, budget - step + 1)]):
            if stop and stop():
                aborted = True
                break
            obs, info, done, c0, c1 = sim.step(a, first=(step == 0))
            ok, step = bool(info["last_action_success"]), info["env_step"]
            src.observe(info, dict(obs, step=step))
            # render the whole clip first, then play it: rendering time varies under GPU load, playback should not
            clip = [panel.frame(sim.show(cam), cam) for cam in sim.motion(c0, c1, a, ok, fps)]
            for img in clip:
                yield ("frame", img, 1.0 / fps)
            traj.append((obs["pose"]["x"], obs["pose"]["z"]))
            status.append("ok" if ok else "rej")
            yield ("step", {"step": step, "action": sim.skills[a], "ok": ok, "distance": round(float(info["distance"]), 2),
                            "calls": src.calls()})
            vlm, dec = src.text(T, reasoning, entry, actions, j + 1, ok)
            panel.set_key(step, status, traj, vlm, dec, hyps, goal, src.occ_map, cur=-1 if done else step)
            yield ("frame", panel.frame(sim.disp, obs["pose"], rejected=not ok), 0.2 if ok else 0.7)
            if done or not ok:
                break
        if aborted:
            break
    success = bool(info.get("task_success", 0) >= 1)
    dist = float(info["distance"]) if info else nan
    panel.set_key(step, status, traj, vlm, dec, hyps, None, src.occ_map, cur=-1)
    yield ("frame", panel.frame(sim.disp, obs["pose"], outcome=(success, dist, aborted)), 3.0)
    yield ("done", {"success": success, "distance": round(dist, 2) if dist == dist else None, "steps": step,
                    "aborted": aborted, "calls": src.calls()})
