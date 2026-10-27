"""RoboTHOR ObjectNav runner for S2M (second benchmark).

  python -m embagent.robothor.runner --model Qwen3-VL-4B-Instruct-AWQ --exp v1 --split dev --per-type 3 --workers 1
  python -m embagent.robothor.runner --model ... --exp final --split val --per-type 25 --workers 2

--per-type N takes a fixed stratified subset (N episodes per target category, seed 0); 0 = the whole split.
Splits: val = official 1800 val episodes (reporting); dev = episodes from RoboTHOR *train* scenes (tuning only).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def select(eps, per_type, seed=0):
    if not per_type:
        return eps
    by = defaultdict(list)
    for e in eps:
        by[e["object_type"]].append(e)
    rng = random.Random(seed)
    out = []
    for t in sorted(by):
        out += rng.sample(by[t], min(per_type, len(by[t])))
    return out


def worker(wid, eps, model, out_dir, planner_kw):
    sys.path.insert(0, ROOT)
    from embagent.planners.s2m_robothor import S2MRoboTHOR
    from embagent.robothor.env import RoboTHOREnv

    env, planner = None, S2MRoboTHOR(model, **planner_kw)
    for ep in eps:
        path = os.path.join(out_dir, f"{ep['id']}.json")
        if os.path.exists(path):
            continue
        rec = None
        for attempt in range(2):
            try:
                if env is None:
                    env = RoboTHOREnv()
                t0 = time.time()
                obs = env.reset(ep)
                planner.reset(ep["object_type"])
                done, modes = False, []
                while not done:
                    a, info = planner.act(obs)
                    modes.append(info.get("mode", "?")[0])
                    obs, done, sinfo = env.step(a)
                    planner.observe(sinfo, obs)
                rec = {"id": ep["id"], "scene": ep["scene"], "object_type": ep["object_type"],
                       "success": float(env.success), "spl": env.spl(), "steps": env.steps,
                       "path_len": round(env.path_len, 2), "final_dist": env.target_distance(),
                       "stopped": modes[-1:] == ["s"] or env.steps < 500, "modes": "".join(modes),
                       "elapsed_s": round(time.time() - t0, 1)}
                rec.update(planner.stats)
                break
            except Exception as e:
                traceback.print_exc()
                rec = {"id": ep["id"], "scene": ep["scene"], "object_type": ep["object_type"], "success": 0.0,
                       "spl": 0.0, "error": repr(e)}
                try:
                    env.close()
                except Exception:
                    pass
                env = None
        with open(path, "w") as f:
            json.dump(rec, f)
        print(f"[w{wid}] {ep['id']} success={rec['success']:.0f} steps={rec.get('steps')} "
              f"dist={rec.get('final_dist')} t={rec.get('elapsed_s')}s", flush=True)
    if env is not None:
        env.close()


def summarize(out_dir):
    recs = [json.load(open(os.path.join(out_dir, f))) for f in os.listdir(out_dir) if f.endswith(".json")]
    by = defaultdict(list)
    for r in recs:
        by[r["object_type"]].append(r)
    rows = []
    for t in sorted(by) + ["ALL"]:
        rs = recs if t == "ALL" else by[t]
        n = len(rs)
        rows.append({"type": t, "n": n, "SR": round(100 * sum(r["success"] for r in rs) / n, 1),
                     "SPL": round(100 * sum(r["spl"] for r in rs) / n, 1),
                     "errors": sum("error" in r for r in rs)})
    json.dump(rows, open(os.path.join(out_dir, "..", os.path.basename(out_dir) + "_summary.json"), "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "val"])
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--s2m", default="{}", help="JSON kwargs for S2MRoboTHOR")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from embagent.robothor.env import load_split

    eps = select(load_split(args.split), args.per_type)
    out_dir = os.path.join(ROOT, "runs", "robothor", args.split, f"s2m_{args.model}_{args.exp}")
    os.makedirs(out_dir, exist_ok=True)
    todo = [e for e in eps if not os.path.exists(os.path.join(out_dir, f"{e['id']}.json"))]
    print(f"{len(eps)} episodes selected, {len(todo)} to run -> {out_dir}", flush=True)
    todo.sort(key=lambda e: e["scene"])  # keep scene resets cheap within a worker
    shards = [todo[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(w, s, args.model, out_dir, json.loads(args.s2m)))
             for w, s in enumerate(shards) if s]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for row in summarize(out_dir):
        print(row)


if __name__ == "__main__":
    main()
