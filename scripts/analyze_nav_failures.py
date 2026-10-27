"""Classify EB-Navigation episode outcomes from nav_runner records (uses privileged sim info, analysis only).

Categories (first match wins):
  success          reached <= 1 m
  never_seen       target never in view during the episode
  near_miss        ended within 1.5 m (close but not close enough)
  lost_after_seen  saw the target, but it was out of view for the last 5 steps
  blocked          >= 30% of actions were invalid (collisions)
  oscillation      >= 3 immediate reversals (forward/back, left/right, rotate L/R)
  approach_fail    saw the target and kept it in view, but did not close the distance

Usage: python scripts/analyze_nav_failures.py runs/eb_nav/<run_dir> [<run_dir> ...]
"""
import collections
import json
import os
import sys

REVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def classify(rec):
    if rec.get("task_success", 0) >= 1:
        return "success"
    traj = rec.get("trajectory") or []
    if len(traj) < 2:
        return "error" if "error" in rec else "no_steps"
    vis = [t.get("vis") for t in traj]
    acts = [t["action"] for t in traj[1:]]
    if not any(vis):
        return "never_seen"
    if (rec.get("final_distance") or 99) <= 1.5:
        return "near_miss"
    if not any(vis[-5:]):
        return "lost_after_seen"
    if sum(not t["ok"] for t in traj[1:]) >= 0.3 * len(acts):
        return "blocked"
    if sum(REVERSE.get(a) == b for a, b in zip(acts, acts[1:])) >= 3:
        return "oscillation"
    return "approach_fail"


def main(run_dirs):
    for run in run_dirs:
        total = collections.Counter()
        per_set = {}
        for eval_set in sorted(os.listdir(run)):
            d = os.path.join(run, eval_set)
            if not os.path.isdir(d) or eval_set.startswith("_"):
                continue
            c = collections.Counter()
            for f in os.listdir(d):
                if f.endswith(".json"):
                    c[classify(json.load(open(os.path.join(d, f))))] += 1
            per_set[eval_set] = c
            total.update(c)
        n = sum(total.values())
        print(f"\n== {run}  ({n} episodes)")
        cats = ["success", "never_seen", "lost_after_seen", "approach_fail", "near_miss", "blocked",
                "oscillation", "error", "no_steps"]
        print(f"{'subset':22}" + "".join(f"{c[:13]:>15}" for c in cats))
        for s, c in list(per_set.items()) + [("TOTAL", total)]:
            m = sum(c.values()) or 1
            print(f"{s:22}" + "".join(f"{c[k]:>6} ({100 * c[k] / m:4.1f}%)" for k in cats))


if __name__ == "__main__":
    main(sys.argv[1:])
