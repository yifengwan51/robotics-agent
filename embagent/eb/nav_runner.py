"""EB-Navigation runner with pluggable planners.

Shares the official evaluator's environment class (EBNavigationEnv), success test (<= 1 m to the target),
action set and 20-step budget, and replans on an invalid action exactly like EB_NavigationEvaluator.
Additions: planners receive depth + pose, episodes are sharded over worker processes, every episode writes
a JSON record (resumable), and trajectories are kept for analysis.

  python -m embagent.eb.nav_runner --planner eb --model Qwen2.5-VL-3B-Instruct-AWQ --exp baseline --workers 2
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_SETS = ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"]


def _depth_model():
    """Name of the model the running depth server serves (recorded so a run cannot silently mix depth models)."""
    from embagent.depth_est import served_model
    return served_model()


# settings that change episode outcomes, normalized for comparison (older args.json files lack some keys)
_RESULT_KEYS = {
    "planner": lambda a: a.get("planner"), "model": lambda a: a.get("model"), "n_shots": lambda a: a.get("n_shots"),
    "resolution": lambda a: a.get("resolution"), "data": lambda a: os.path.abspath(a["data"]) if a.get("data") else "",
    "s2m": lambda a: json.loads(a.get("s2m") or "{}"), "depth": lambda a: a.get("depth", "gt"),
    "odom_noise": lambda a: [float(x) for x in a.get("odom_noise", (0.0, 0.0))], "repeat": lambda a: a.get("repeat", 0),
    "vlm_extra_body": lambda a: a.get("vlm_extra_body", {}), "vlm_max_tokens": lambda a: a.get("vlm_max_tokens", 4096),
    "vlm_schema_bounds": lambda a: a.get("vlm_schema_bounds", {}), "odom": lambda a: a.get("odom", "sim"),
    "depth_model": lambda a: a.get("depth_model") or ("DA2-Metric-Indoor-Small" if a.get("depth") == "est" else ""),
    "depth_calib": lambda a: a.get("depth_calib") or ("depth_calib.json" if a.get("depth") == "est" else ""),
}


def _record_path(out_dir, eval_set, idx):
    return os.path.join(out_dir, eval_set, f"episode_{idx + 1}.json")


def _pose(event):
    a = event.metadata["agent"]
    return {"x": a["position"]["x"], "z": a["position"]["z"], "yaw": a["rotation"]["y"],
            "horizon": event.metadata.get("agent", {}).get("cameraHorizon", 0.0)}


def _target_state(env):
    """Privileged target info for ANALYSIS ONLY (never passed to planners): distance and in-view flag."""
    ev = env.env.last_event
    tid = env.episode_data.get("targetObjectIds")
    vis = next((o["visible"] for o in ev.metadata["objects"] if o["objectId"] == tid), None)
    _, dist = env.measure_success()
    return {"dist": round(dist, 3), "vis": vis}


def _obs(env, step, oracle=False):
    ev = env.env.last_event
    obs = {"rgb": ev.frame, "depth": ev.depth_frame, "pose": _pose(ev),
           "instruction": env.episode_language_instruction, "step": step}
    if oracle:  # diagnostic planners only
        t = env.episode_data["target_position"]
        obs["oracle_target"] = (t["x"], t["z"])
    return obs


class Sensors:
    """What the planner perceives: ground-truth or estimated depth, exact or drifting odometry.

    Odometry noise integrates the true per-step displacement with zero-mean Gaussian error
    (sigma_t * |translation| in metres, sigma_r degrees per rotation), so errors accumulate like real wheel/VO
    odometry. The evaluation (success, logged distance) always uses the true simulator state."""

    # odom="actions": pose by dead reckoning from the executed action and its success flag only (the same feedback
    # the EB-prompt baseline gets), starting from the episode's start pose; the simulator's 0.1 m grid snapping is
    # not modelled, so the estimate drifts by up to one cell per translation
    MOVES = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # action id -> (forward, right) in 0.25 m steps

    def __init__(self, depth="gt", odom_noise=(0.0, 0.0), seed=0, odom="sim"):
        import numpy as np

        self.depth, (self.sig_t, self.sig_r), self.odom = depth, odom_noise, odom
        assert not (odom == "actions" and (self.sig_t > 0 or self.sig_r > 0)), "--odom actions excludes --odom-noise"
        self.rng = np.random.default_rng(seed)
        self.true_prev = self.est = None
        self.last = None  # (action, success) of the step that produced the next observation

    def record(self, action, ok):
        self.last = (action, ok)

    def _dead_reckon(self, t):
        import math

        if self.est is None:
            self.est = dict(t)
        elif self.last is not None and self.last[1]:
            a, e = self.last[0], self.est
            if a in self.MOVES:
                fwd, rgt = (0.25 * v for v in self.MOVES[a])
                y = math.radians(e["yaw"])
                e["x"] += fwd * math.sin(y) + rgt * math.cos(y)
                e["z"] += fwd * math.cos(y) - rgt * math.sin(y)
            elif a in (4, 5):
                e["yaw"] = (e["yaw"] + (90 if a == 4 else -90)) % 360
            elif a in (6, 7):
                e["horizon"] += -30 if a == 6 else 30
        self.last = None
        return dict(self.est)

    def __call__(self, obs):
        import math

        if self.depth == "est":
            from embagent.depth_est import estimate
            obs["depth"] = estimate(obs["rgb"])
        if self.odom == "actions":
            obs["true_pose"] = obs["pose"]
            obs["pose"] = self._dead_reckon(obs["pose"])
            return obs
        if self.sig_t > 0 or self.sig_r > 0:
            t = obs["pose"]
            if self.est is None:
                self.est = dict(t)
            else:
                p = self.true_prev
                dx, dz, dyaw = t["x"] - p["x"], t["z"] - p["z"], (t["yaw"] - p["yaw"] + 180) % 360 - 180
                n = math.hypot(dx, dz)
                # express the true motion in the previous estimated heading frame, then corrupt it
                fwd = dx * math.sin(math.radians(p["yaw"])) + dz * math.cos(math.radians(p["yaw"]))
                rgt = dx * math.cos(math.radians(p["yaw"])) - dz * math.sin(math.radians(p["yaw"]))
                if n > 1e-3:
                    fwd += self.rng.normal(0, self.sig_t * n)
                    rgt += self.rng.normal(0, self.sig_t * n)
                if abs(dyaw) > 1e-3:
                    dyaw += self.rng.normal(0, self.sig_r)
                y = math.radians(self.est["yaw"])
                self.est = {"x": self.est["x"] + fwd * math.sin(y) + rgt * math.cos(y),
                            "z": self.est["z"] + fwd * math.cos(y) - rgt * math.sin(y),
                            "yaw": (self.est["yaw"] + dyaw) % 360, "horizon": t["horizon"]}
            self.true_prev = dict(t)
            obs["true_pose"], obs["pose"] = t, dict(self.est)
        return obs


def run_episode(env, planner, max_retries_per_call=3, sensors=None):
    import numpy as np

    oracle = getattr(planner, "needs_oracle", False)
    sense = sensors or (lambda o: o)
    env.reset()
    planner.reset(env.episode_language_instruction)
    obs = sense(_obs(env, 0, oracle))
    traj = [dict(obs.get("true_pose", obs["pose"]), action=None, ok=True, **_target_state(env))]
    plans, done, info = [], False, {"task_success": 0.0, "distance": None, "env_step": 0}
    t0 = time.time()
    while not done:
        actions, reasoning = planner.act(obs)
        if not actions:  # an empty plan would never advance the episode
            actions = [int(np.random.randint(len(env.language_skill_set)))]
        plans.append({"step": env._current_step, "actions": actions,
                      "reasoning": {k: (v if isinstance(v, (int, float, bool, list, type(None))) else str(v)[:600])
                                    for k, v in (reasoning or {}).items()}})
        budget = env._max_episode_steps - env._current_step + 1
        for i, a in enumerate(actions[:max(1, budget)]):
            if not isinstance(a, int) or not 0 <= a < len(env.language_skill_set):
                a = int(np.random.randint(len(env.language_skill_set)))  # same fallback as the EB env
            _, _, done, info = env.step(a, reasoning, 1 if i == 0 else 0)
            if hasattr(sense, "record"):
                sense.record(a, bool(info["last_action_success"]))
            obs = sense(_obs(env, env._current_step, oracle))
            planner.observe(info, obs)
            traj.append(dict(obs.get("true_pose", obs["pose"]), action=a, ok=bool(info["last_action_success"]), **_target_state(env)))
            if done or not info["last_action_success"]:
                break  # invalid action -> replan (EB_NavigationEvaluator behaviour)
    rec = {"instruction": env.episode_language_instruction, "task_success": float(info["task_success"]),
           "final_distance": info.get("distance"), "num_steps": info["env_step"], "planner_calls": len(plans),
           "elapsed_s": round(time.time() - t0, 1), "trajectory": traj, "plans": plans,
           "target": env.episode_data.get("target_position"), "scene": env.episode_data.get("scene")}
    rec.update(getattr(planner, "stats", {}))
    return rec


def worker(wid, jobs, args_dict, out_dir):
    sys.path.insert(0, ROOT)
    from embagent.eb import shim

    shim.install()
    import numpy as np
    from embodiedbench.envs.eb_navigation import EBNavEnv
    from embagent.planners import make_planner
    from embagent.eb.local_model import STATS

    if args_dict.get("platform") == "linux64":
        from ai2thor.platform import Linux64
        EBNavEnv.CloudRendering = Linux64

    planner = make_planner(args_dict["planner"], EBNavEnv.DISCRETE_SKILLSET, **args_dict)
    by_set = {}
    for eval_set, idx in jobs:
        by_set.setdefault(eval_set, []).append(idx)

    for eval_set, idxs in by_set.items():
        env, full = None, None
        for idx in idxs:
            path = _record_path(out_dir, eval_set, idx)
            for attempt in range(4):
                if attempt:  # renderer crashes are usually transient VRAM pressure: back off before restarting
                    import time as _t
                    _t.sleep(20 * attempt)
                try:
                    if env is None:  # one controller per subset; episodes are swapped in below
                        env = EBNavEnv.EBNavigationEnv(eval_set=eval_set, exp_name=f"w{wid}",
                                                       resolution=args_dict["resolution"], selected_indexes=[0])
                        env.log_path = os.path.join(out_dir, "_eb_logs", eval_set, f"w{wid}")
                        if args_dict.get("data"):
                            env.data_path = os.path.join(args_dict["data"], f"{eval_set}.json")
                        full = env._load_dataset(eval_set)
                    env.dataset, env.selected_indexes = [full[idx]], [idx]
                    env.number_of_episodes, env._current_episode_num = 1, 0
                    seed = int(hashlib.md5(f"{eval_set}-{idx}".encode()).hexdigest()[:8], 16)
                    np.random.seed(seed)
                    STATS.reset()
                    sensors = Sensors(args_dict.get("depth", "gt"), tuple(args_dict.get("odom_noise", (0, 0))),
                                      odom=args_dict.get("odom", "sim"),
                                      seed=seed + 7919 * args_dict.get("repeat", 0))
                    rec = run_episode(env, planner, sensors=sensors)
                    rec.update(eval_set=eval_set, index=idx, model_calls=STATS.snapshot())
                    break
                except Exception as e:
                    traceback.print_exc()
                    rec = {"eval_set": eval_set, "index": idx, "task_success": 0.0, "error": repr(e)}
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(rec, f)
            print(f"[w{wid}] {eval_set}#{idx + 1} success={rec['task_success']:.0f} "
                  f"steps={rec.get('num_steps')} dist={rec.get('final_distance')}", flush=True)
        if env is not None:
            env.close()


def summarize(out_dir):
    rows = []
    for eval_set in EVAL_SETS:
        d = os.path.join(out_dir, eval_set)
        if not os.path.isdir(d):
            continue
        recs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]
        if not recs:
            continue
        sr = sum(r["task_success"] for r in recs) / len(recs)
        rows.append({"eval_set": eval_set, "n": len(recs), "success_rate": round(100 * sr, 1),
                     "avg_steps": round(sum(r.get("num_steps") or 0 for r in recs) / len(recs), 2),
                     "errors": sum(1 for r in recs if "error" in r),
                     "json_errors": sum(r.get("json_errors", 0) for r in recs)})
    if rows:
        n = sum(r["n"] for r in rows)
        overall = sum(r["success_rate"] * r["n"] for r in rows) / n
        rows.append({"eval_set": "AVG(subsets)", "n": n,
                     "success_rate": round(sum(r["success_rate"] for r in rows) / len(rows), 1),
                     "pooled_success_rate": round(overall, 1)})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="eb")
    ap.add_argument("--model", default="Qwen2.5-VL-3B-Instruct-AWQ")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--eval-sets", nargs="*", default=EVAL_SETS)
    ap.add_argument("--limit", type=int, default=0, help="episodes per subset (0 = all 60)")
    ap.add_argument("--stride", type=int, default=1, help="take every k-th episode")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=500)
    ap.add_argument("--n-shots", type=int, default=3)
    ap.add_argument("--platform", default="cloud", choices=["cloud", "linux64"])
    ap.add_argument("--data", default="", help="dataset dir (default: official EB test set); e.g. data/eb_nav_dev")
    ap.add_argument("--s2m", default="{}", help='JSON kwargs for the s2m planner, e.g. \'{"commit": 1.2}\'')
    ap.add_argument("--depth", default="gt", choices=["gt", "est"], help="est: monocular depth (embagent/depth_est.py)")
    ap.add_argument("--odom-noise", nargs=2, type=float, default=[0.0, 0.0], metavar=("SIG_T", "SIG_R_DEG"))
    ap.add_argument("--odom", default="sim", choices=["sim", "actions"],
                    help="actions: pose by dead reckoning from executed actions and their success (RGB-only setting)")
    ap.add_argument("--repeat", type=int, default=0, help="repeat index; changes noise seeds, nothing else")
    ap.add_argument("--retry-errors", action="store_true", help="re-run episodes whose record has an error")
    ap.add_argument("--allow-subset", action="store_true", help="permit --limit/--stride on the test split (debug only)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "eb_nav"))
    args = ap.parse_args()

    split = "dev" if args.data else "test"
    if split == "test" and (args.limit or args.stride != 1) and not args.allow_subset:
        sys.exit("Formal runs must use the full official EB-Navigation test set: drop --limit/--stride "
                 "(or pass --allow-subset for a debug run that must not be reported).")
    if split == "test" and sorted(args.eval_sets) != sorted(EVAL_SETS):
        print(f"WARNING: running only {args.eval_sets}; a reportable result needs all five subsets", flush=True)
    out_dir = os.path.join(args.out, split, f"{args.planner}_{args.model.split('/')[-1]}_{args.exp}")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.abspath(args.data) if args.data else os.path.join(
        ROOT, "third_party/EmbodiedBench/embodiedbench/envs/eb_navigation/datasets")
    jobs = []
    for eval_set in args.eval_sets:
        n_ep = len(json.load(open(os.path.join(data_dir, f"{eval_set}.json")))["tasks"])
        idxs = list(range(0, n_ep, args.stride))
        if args.limit:
            idxs = idxs[:args.limit]
        def pending(i):
            path = _record_path(out_dir, eval_set, i)
            if not os.path.exists(path):
                return True
            return args.retry_errors and "error" in json.load(open(path))
        jobs += [(eval_set, i) for i in idxs if pending(i)]
    print(f"{len(jobs)} episodes to run -> {out_dir}", flush=True)
    # args.json records the settings the episodes were produced with. A resume (e.g. --retry-errors) must not
    # rewrite it, and must not run new episodes under settings different from the ones already recorded.
    record = dict(vars(args), vlm_extra_body=json.loads(os.environ.get("VLM_EXTRA_BODY") or "{}"),
                  vlm_max_tokens=int(os.environ.get("VLM_MAX_TOKENS", "4096")),
                  vlm_schema_bounds=json.loads(os.environ.get("VLM_SCHEMA_BOUNDS") or "{}"),
                  depth_model=_depth_model() if args.depth == "est" else "",
                  depth_calib=os.path.basename(os.environ.get("DEPTH_CALIB") or "depth_calib.json") if args.depth == "est" else "")
    args_path = os.path.join(out_dir, "args.json")
    if os.path.exists(args_path):
        mismatch = {k: (old, new) for k, (old, new) in
                    ((k, (v(json.load(open(args_path))), v(record))) for k, v in _RESULT_KEYS.items()) if old != new}
        if mismatch and jobs:
            sys.exit(f"{args_path} was recorded with different settings {mismatch} (recorded, now); refusing to mix "
                     "configurations in one run directory. Use a new --exp or fix the flags.")
    if jobs:
        with open(args_path, "w") as f:
            json.dump(record, f, indent=1)

    planner_kw = dict(planner=args.planner, model_name=args.model, n_shots=args.n_shots,
                      resolution=args.resolution, platform=args.platform,
                      data=os.path.abspath(args.data) if args.data else "", s2m=json.loads(args.s2m),
                      depth=args.depth, odom_noise=args.odom_noise, repeat=args.repeat, odom=args.odom)
    shards = [jobs[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(w, s, planner_kw, out_dir)) for w, s in enumerate(shards) if s]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for row in summarize(out_dir):
        print(row)


if __name__ == "__main__":
    main()
