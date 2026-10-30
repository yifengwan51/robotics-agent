"""Read-only diagnostics for the paper, computed from saved EB-Navigation TEST records (no model or simulator).

  python scripts/paper_diagnostics.py      -> stdout and paper/tables/diagnostics.md

1. SR with bootstrap 95% CI for every formal run.
2. Paired exact McNemar tests on the 300 shared episodes (EB prompt vs S2M; S2M run 1 vs each variant), with a
   Holm correction over the nine single-switch variants.
3. Failure taxonomy (scripts/analyze_nav_failures.py) per run.
4. S2M candidate recall: does any logged detection/hypothesis lie within R of the true target (x-z plane)?
   Exact-pose runs only; the perception log covers all <= 20 steps.
5. Qwen2.5 EB-prompt parse failures: affected calls/episodes, implied completion tokens per failed call, and the
   decode-error messages in logs/queue_25.log.
6. Crop-verification probe (runs/grounding/verify_*_test.jsonl), same definitions as probe_verification.report.
Privileged target state is used here for analysis only.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_nav_failures import classify  # noqa: E402
from stats import SUBSETS, boot_ci, load, mcnemar  # noqa: E402

TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
OUT = os.path.join(ROOT, "paper", "tables", "diagnostics.md")
Q3, Q25 = "Qwen3-VL-4B-Instruct-AWQ", "Qwen2.5-VL-3B-Instruct-AWQ"
SHORT = {"base": "Base", "common_sense": "Common", "complex_instruction": "Complex",
         "visual_appearance": "Visual", "long_horizon": "Long"}
RUNS = [("EB prompt, Qwen3", f"eb_{Q3}_eb_official_r1"), ("S2M r1, Qwen3", f"s2m_{Q3}_final_r1"),
        ("S2M r2, Qwen3", f"s2m_{Q3}_final_r2"), ("S2M r3, Qwen3", f"s2m_{Q3}_final_r3"),
        ("EB prompt, Qwen2.5", f"eb_{Q25}_eb_official_r1"), ("S2M, Qwen2.5", f"s2m_{Q25}_final_r1"),
        ("Oracle geometry", "oracle_geo_none_ceiling"),
        ("EB prompt, Qwen2.5, bounded schema", f"eb_{Q25}_eb_bounded_r1")]
VARIANTS = [("no negative evidence", "abl_no_negative"), ("no multi-view fusion", "abl_no_fusion"),
            ("no tiled search", "abl_no_tiles"), ("no crop verification", "abl_no_zoom"),
            ("no look-around", "abl_no_scan"), ("no exhaust-on-arrival", "abl_no_exhaust"),
            ("strongest-first", "abl_strongest_first"), ("estimated depth", "depth_est_r1"),
            ("odometry noise", "odom_noise_r1")]
CATS = ["success", "never_seen", "lost_after_seen", "approach_fail", "near_miss", "blocked", "oscillation"]
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def table(head, rows):
    out("| " + " | ".join(head) + " |")
    out("|" + "---|" * len(head))
    for r in rows:
        out("| " + " | ".join(map(str, r)) + " |")
    out()


def records(run):
    recs = {}
    for s in SUBSETS:
        for f in glob.glob(os.path.join(TEST, run, s, "episode_*.json")):
            recs[(s, os.path.basename(f))] = json.load(open(f))
    return recs


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def paired_row(a, b):
    da, db = load(os.path.join(TEST, a)), load(os.path.join(TEST, b))
    ks = sorted(set(da) & set(db))
    n01, n10, p = mcnemar([da[k] for k in ks], [db[k] for k in ks])
    m, lo, hi = boot_ci([db[k] - da[k] for k in ks])
    return len(ks), m, lo, hi, n01, n10, p


def success_rates():
    out("## 1. Success rate with bootstrap 95% CI (300 episodes each)\n")
    rows = []
    reps = [load(os.path.join(TEST, f"s2m_{Q3}_final_r{i}")) for i in (1, 2, 3)]
    ks = sorted(set.intersection(*[set(d) for d in reps]))
    m, lo, hi = boot_ci([np.mean([d[k] for d in reps]) for k in ks])
    rows.append(["S2M 3-run mean, Qwen3 (per-episode mean)", len(ks), f"{m:.1f} [{lo:.1f}, {hi:.1f}]"])
    for label, run in RUNS + [(f"S2M {v}", f"s2m_{Q3}_{d}") for v, d in VARIANTS]:
        d = load(os.path.join(TEST, run))
        m, lo, hi = boot_ci([d[k] for k in sorted(d)])  # same episode order as stats.py
        rows.append([label, len(d), f"{m:.1f} [{lo:.1f}, {hi:.1f}]"])
    table(["Run", "n", "SR (95% CI)"], rows)


def paired_tests():
    out("## 2. Paired comparisons (exact McNemar on shared episodes)\n")
    rows = []
    for label, a, b in [("Qwen3: S2M r1 - EB prompt", f"eb_{Q3}_eb_official_r1", f"s2m_{Q3}_final_r1"),
                        ("Qwen2.5: S2M - EB prompt", f"eb_{Q25}_eb_official_r1", f"s2m_{Q25}_final_r1"),
                        ("Qwen2.5: S2M - EB prompt (bounded schema)", f"eb_{Q25}_eb_bounded_r1", f"s2m_{Q25}_final_r1"),
                        ("Qwen2.5: EB bounded - EB original schema", f"eb_{Q25}_eb_official_r1", f"eb_{Q25}_eb_bounded_r1"),
                        ("S2M r2 - r1", f"s2m_{Q3}_final_r1", f"s2m_{Q3}_final_r2"),
                        ("S2M r3 - r1", f"s2m_{Q3}_final_r1", f"s2m_{Q3}_final_r3")]:
        n, m, lo, hi, n01, n10, p = paired_row(a, b)
        rows.append([label, n, f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]", f"{n01}/{n10}", f"{p:.2g}"])
    table(["Comparison", "n", "diff pp (95% CI)", "B-only/A-only wins", "McNemar p"], rows)

    res = [paired_row(f"s2m_{Q3}_final_r1", f"s2m_{Q3}_{d}") for _, d in VARIANTS]
    adj = holm([r[6] for r in res])
    rows = [[v, f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]", f"{n01}/{n10}", f"{p:.2g}", f"{q:.2g}"]
            for (v, _), (_, m, lo, hi, n01, n10, p), q in zip(VARIANTS, res, adj)]
    table(["Variant vs S2M r1", "diff pp (95% CI)", "variant-only/r1-only wins", "McNemar p", "Holm p (9 tests)"],
          rows)

    out("Per-subset paired differences (pp; McNemar p in parentheses):\n")
    rows = []
    for label, a, b in [("S2M r1 - EB prompt (Qwen3)", f"eb_{Q3}_eb_official_r1", f"s2m_{Q3}_final_r1")] + \
            [(v, f"s2m_{Q3}_final_r1", f"s2m_{Q3}_{d}") for v, d in VARIANTS]:
        da, db = load(os.path.join(TEST, a)), load(os.path.join(TEST, b))
        cells = []
        for s in SUBSETS:
            ks = sorted(k for k in set(da) & set(db) if k[0] == s)
            _, _, p = mcnemar([da[k] for k in ks], [db[k] for k in ks])
            cells.append(f"{100 * (np.mean([db[k] for k in ks]) - np.mean([da[k] for k in ks])):+.1f} ({p:.2g})")
        rows.append([label] + cells)
    table(["Comparison"] + [SHORT[s] for s in SUBSETS], rows)


def taxonomy():
    out("## 3. Outcome taxonomy (episodes; scripts/analyze_nav_failures.py)\n")
    rows = []
    for label, run in RUNS[:2] + RUNS[4:6] + RUNS[7:8] + [("S2M no fusion", f"s2m_{Q3}_abl_no_fusion"),
                                             ("S2M no look-around", f"s2m_{Q3}_abl_no_scan"),
                                             ("S2M estimated depth", f"s2m_{Q3}_depth_est_r1")]:
        c = collections.Counter(classify(r) for r in records(run).values())
        rows.append([label] + [c[k] for k in CATS] + [sum(c.values()) - sum(c[k] for k in CATS)])
    table(["Run"] + CATS + ["other"], rows)
    out("Long-horizon subset only:\n")
    rows = []
    for label, run in RUNS[:2] + [("S2M no look-around", f"s2m_{Q3}_abl_no_scan")]:
        c = collections.Counter(classify(r) for (s, _), r in records(run).items() if s == "long_horizon")
        rows.append([label] + [c[k] for k in CATS])
    table(["Run"] + CATS, rows)


def candidate_recall():
    out("## 4. S2M candidate recall (any logged detection or hypothesis within R of the target, x-z plane)\n")
    rows = []
    for label, run in [("S2M r1, Qwen3", f"s2m_{Q3}_final_r1"), ("S2M r2, Qwen3", f"s2m_{Q3}_final_r2"),
                       ("S2M r3, Qwen3", f"s2m_{Q3}_final_r3"), ("S2M, Qwen2.5", f"s2m_{Q25}_final_r1")]:
        recs = records(run)
        for radius in (0.5, 1.0):
            cells, fail_split, vis_split = [], [0, 0], [0, 0]
            for s in SUBSETS + ["ALL"]:
                hit = []
                for (sub, _), r in recs.items():
                    if s not in ("ALL", sub):
                        continue
                    t = np.array([r["target"]["x"], r["target"]["z"]])
                    pts = [p[:2] for e in r["perception_log"] for p in e["dets"] + e["hyps"]]
                    h = any(np.hypot(*(np.array(p) - t)) <= radius for p in pts)
                    hit.append(h)
                    if s == "ALL" and r["task_success"] < 1:
                        fail_split[int(h)] += 1
                        if sub == "visual_appearance":
                            vis_split[int(h)] += 1
                cells.append(f"{100 * np.mean(hit):.1f}")
            rows.append([label, radius] + cells + [f"{fail_split[0]}/{fail_split[1]}",
                                                   f"{vis_split[0]}/{vis_split[1]}"])
    table(["Run", "R (m)"] + [SHORT[s] for s in SUBSETS] +
          ["All", "failures: no cand./cand.", "Visual failures: no cand./cand."], rows)


def qwen25_parse_failures():
    out("## 5. Qwen2.5 EB-prompt parse failures\n")
    for label, run in [("Qwen2.5", f"eb_{Q25}_eb_official_r1"), ("Qwen2.5 bounded schema", f"eb_{Q25}_eb_bounded_r1"),
                       ("Qwen3", f"eb_{Q3}_eb_official_r1")]:
        recs = list(records(run).values())
        calls = sum(r["model_calls"]["calls"] for r in recs)
        errs = np.array([r.get("json_errors", 0) for r in recs])
        comp = np.array([r["model_calls"]["completion_tokens"] for r in recs], float)
        ncall = np.array([r["model_calls"]["calls"] for r in recs], float)
        succ = np.array([r["task_success"] >= 1 for r in recs])
        out(f"- {label}: {calls} planner calls, {errs.sum()} decode failures ({100 * errs.sum() / calls:.1f}% of "
            f"calls), {int((errs > 0).sum())}/{len(recs)} episodes affected; mean completion tokens/call "
            f"{comp.sum() / calls:.0f}")
        if errs.sum():
            # completion tokens per episode = a * clean calls + b * failed calls (least squares, no intercept)
            (a, b), *_ = np.linalg.lstsq(np.stack([ncall - errs, errs], 1), comp, rcond=None)
            out(f"  - fitted completion tokens: {a:.0f} per parsed call, {b:.0f} per failed call "
                f"(budget VLM_MAX_TOKENS = 4096); failed calls account for {100 * b * errs.sum() / comp.sum():.1f}% "
                f"of all completion tokens")
            out(f"  - SR: episodes with >= 1 failure {100 * succ[errs > 0].mean():.1f}% "
                f"(n={int((errs > 0).sum())}), without {100 * succ[errs == 0].mean():.1f}% (n={int((errs == 0).sum())})")
    log = os.path.join(ROOT, "logs", "queue_25.log")
    msgs = collections.Counter()
    for line in open(log, errors="replace"):
        m = re.search(r"Failed to decode JSON: ([^:]+): .*\(char (\d+)\)", line)
        if m:
            msgs[(m.group(1), "char >= 10000" if int(m.group(2)) >= 10000 else "char < 10000")] += 1
    other = collections.Counter()
    for line in open(log, errors="replace"):
        for pat in ("An unexpected error occurred", "empty plan", "[LocalOpenAIModel] call failed"):
            other[pat] += pat in line
    out(f"- other fallbacks in logs/queue_25.log: " + ", ".join(f"'{k}' {v}" for k, v in other.items()))
    out()
    table(["decode error (logs/queue_25.log)", "position", "count"], [[k[0], k[1], v] for k, v in msgs.most_common()])


def verification():
    out("## 6. Crop-verification probe, TEST (frames with the target visible and >= 2 candidates)\n")
    path = os.path.join(ROOT, "runs", "grounding", f"verify_{Q3}_test.jsonl")
    recs = [json.loads(line) for line in open(path)]

    def auc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))

    rows = []
    for sub in sorted({r["subset"] for r in recs}) + ["ALL"]:
        rs = [r for r in recs if sub in ("ALL", r["subset"])]
        with_true = [r for r in rs if any(r.get("is_true", []))]
        multi = [r for r in with_true if r["n_cand"] >= 2]
        pos_a, neg_a, pos_r, neg_r = [], [], [], []
        for r in multi:
            for t, a, rel in zip(r["is_true"], r["abs"], r["rel"]):
                (pos_a if t else neg_a).append(a)
                (pos_r if t else neg_r).append(rel)
        top_a = np.mean([r["is_true"][int(np.argmax(r["abs"]))] for r in multi])
        top_r = np.mean([r["is_true"][int(np.argmax(r["rel"]))] for r in multi])
        rows.append([sub, len(rs), f"{np.mean([r['single_hit'] for r in rs]):.2f}",
                     f"{len(with_true) / len(rs):.2f}", len(multi), f"{top_a:.2f}", f"{top_r:.2f}",
                     f"{auc(pos_a, neg_a):.2f}", f"{auc(pos_r, neg_r):.2f}"])
    table(["subset", "visible frames", "single-box hit", "target in candidates", "multi-cand. frames",
           "top-1 crop", "top-1 SoM", "AUC crop", "AUC SoM"], rows)


if __name__ == "__main__":
    out("# Paper diagnostics (generated by scripts/paper_diagnostics.py from saved TEST records)\n")
    success_rates()
    paired_tests()
    taxonomy()
    candidate_recall()
    qwen25_parse_failures()
    verification()
    out("## 7. RoboTHOR val (bootstrap 95% CI over episodes)\n")
    robo = [json.load(open(f)) for f in sorted(glob.glob(
        os.path.join(ROOT, "runs", "robothor", "val", f"s2m_{Q3}_final", "*.json")))]
    for key in ("success", "spl"):
        m, lo, hi = boot_ci([r[key] for r in robo])
        out(f"- {key}: {m:.1f} [{lo:.1f}, {hi:.1f}] (n={len(robo)})")
    with open(OUT, "w") as fp:
        fp.write("\n".join(LINES) + "\n")
