"""Success-rate statistics for nav_runner result directories.

  python scripts/stats.py RUN_DIR [RUN_DIR ...]            per-run SR with bootstrap 95% CI
  python scripts/stats.py --paired RUN_A RUN_B             paired comparison on shared episodes (McNemar)

A run with repeats (RUN_DIR_r1, RUN_DIR_r2, ...) can be given as several directories; per-episode success is
then averaged over repeats before bootstrapping, and mean +- std across repeats is reported too.
"""
import argparse
import json
import math
import os

import numpy as np

SUBSETS = ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"]


def load(run):
    out = {}
    for s in SUBSETS:
        d = os.path.join(run, s)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".json"):
                out[(s, f)] = float(json.load(open(os.path.join(d, f)))["task_success"] >= 1)
    return out


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return 100 * x.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x < y)
    n10 = sum(1 for x, y in zip(a, b) if x > y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def report(runs):
    data = [load(r) for r in runs]
    keys = sorted(set.intersection(*[set(d) for d in data]))
    print(f"{len(runs)} run(s), {len(keys)} shared episodes")
    for s in SUBSETS + ["ALL"]:
        ks = [k for k in keys if s in ("ALL", k[0])]
        if not ks:
            continue
        per_ep = [np.mean([d[k] for d in data]) for k in ks]
        m, lo, hi = boot_ci(per_ep)
        reps = [100 * np.mean([d[k] for k in ks]) for d in data]
        extra = f"  repeats {np.mean(reps):.1f} +- {np.std(reps):.1f}" if len(runs) > 1 else ""
        print(f"  {s:20} n={len(ks):3}  SR {m:5.1f}  95% CI [{lo:5.1f}, {hi:5.1f}]{extra}")


def paired(a, b):
    da, db = load(a), load(b)
    keys = sorted(set(da) & set(db))
    print(f"paired on {len(keys)} episodes: A={a}\n  B={b}")
    for s in SUBSETS + ["ALL"]:
        ks = [k for k in keys if s in ("ALL", k[0])]
        if not ks:
            continue
        xa, xb = [da[k] for k in ks], [db[k] for k in ks]
        n01, n10, p = mcnemar(xa, xb)
        diff = 100 * (np.mean(xb) - np.mean(xa))
        print(f"  {s:20} n={len(ks):3}  A {100 * np.mean(xa):5.1f}  B {100 * np.mean(xb):5.1f}  "
              f"diff {diff:+5.1f}  (B-only wins {n01}, A-only wins {n10})  McNemar p={p:.2g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--paired", action="store_true")
    args = ap.parse_args()
    if args.paired:
        assert len(args.runs) == 2, "--paired takes exactly two run directories"
        paired(*args.runs)
    else:
        report(args.runs)
