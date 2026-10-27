"""Build the paper's result tables from finished runs on the FULL official splits only (EB-Nav test 300, RoboTHOR val 1800) (EB-Navigation) and RoboTHOR runs.

  python scripts/make_tables.py            -> paper/tables/*.md and *.tex (missing runs are marked '--')

Main table: SR per subset + average, mean over repeats (r1..r3) with 95% bootstrap CI; paired McNemar of S2M vs the
same model's EB official-prompt baseline. Ablation table: each one-switch variant vs final_r1 (paired).
Published reference numbers are copied from the cited papers (EB-Nav test split) and marked as such.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from stats import SUBSETS, boot_ci, load, mcnemar  # noqa: E402

TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
OUT = os.path.join(ROOT, "paper", "tables")
SHORT = {"base": "Base", "common_sense": "Common", "complex_instruction": "Complex",
         "visual_appearance": "Visual", "long_horizon": "Long"}
# Published EB-Navigation test numbers (EmbodiedBench Table, ICML 2025; EMHO held-out, arXiv 2610.08432).
PUBLISHED = [
    ("GPT-4o (API)", "EmbodiedBench", None, 57.7),
    ("Qwen2.5-VL-7B (API)", "EmbodiedBench", None, 20.3),
    ("InternVL3-8B", "EmbodiedBench", None, 29.3),
    ("Qwen3.5-9B + EMHO (held-out)", "EMHO", [66.7, 60.0, 65.0, 40.0, 51.7], None),
]
MODELS = ["Qwen3-VL-4B-Instruct-AWQ", "Qwen2.5-VL-3B-Instruct-AWQ", "Qwen3.5-4B-AWQ"]
ABLATIONS = [("abl_no_negative", "w/o negative evidence"), ("abl_no_fusion", "w/o multi-view fusion"),
             ("abl_no_tiles", "w/o magnified tiled search"), ("abl_no_zoom", "w/o crop verification"),
             ("abl_no_scan", "w/o look-around"), ("abl_no_exhaust", "w/o exhaust-on-arrival"),
             ("abl_strongest_first", "strongest-first (no search index)"),
             ("depth_est_r1", "estimated depth (DA-V2)"), ("odom_noise_r1", "noisy odometry (5%, 2 deg)")]


def run_dir(planner, model, exp):
    return os.path.join(TEST, f"{planner}_{model}_{exp}")


def complete(d, n=60):
    return all(os.path.isdir(os.path.join(d, s)) and len(glob.glob(os.path.join(d, s, "*.json"))) >= n for s in SUBSETS)


def per_subset(dirs):
    """SR per subset (mean over repeats) and average over subsets; None where data is missing."""
    data = [load(d) for d in dirs if os.path.isdir(d)]
    if not data:
        return None
    out = {}
    for s in SUBSETS:
        ks = sorted(set.intersection(*[{k for k in d if k[0] == s} for d in data]))
        out[s] = 100 * np.mean([np.mean([d[k] for d in data]) for k in ks]) if len(ks) >= 60 else None  # full subsets only
    vals = [v for v in out.values() if v is not None]
    out["avg"] = float(np.mean(vals)) if len(vals) == len(SUBSETS) else None
    reps = [100 * np.mean([d[k] for k in d]) for d in data if len(d) >= 300]
    out["std"] = float(np.std(reps)) if len(reps) > 1 else None
    return out


def fmt(v, std=None):
    if v is None:
        return "--"
    return f"{v:.1f}" + (f" ± {std:.1f}" if std is not None else "")


def main_table():
    rows = []
    for name, src, subs, avg in PUBLISHED:
        cells = [fmt(x) for x in subs] if subs else ["--"] * 5
        rows.append([f"{name} [{src}]"] + cells + [fmt(avg if avg is not None else (np.mean(subs) if subs else None))])
    for m in MODELS:
        eb = per_subset([run_dir("eb", m, "eb_official_r1")])
        s2m = per_subset([run_dir("s2m", m, f"final_r{i}") for i in (1, 2, 3)])
        for label, r in [(f"{m} + EB prompt", eb), (f"{m} + S2M (ours)", s2m)]:
            if r is None:
                rows.append([label] + ["--"] * 6)
            else:
                rows.append([label] + [fmt(r[s]) for s in SUBSETS] + [fmt(r["avg"], r.get("std"))])
    head = ["Method"] + [SHORT[s] for s in SUBSETS] + ["Avg"]
    return head, rows


def paired_rows():
    rows = []
    for m in MODELS:
        a, b = run_dir("eb", m, "eb_official_r1"), run_dir("s2m", m, "final_r1")
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        da, db = load(a), load(b)
        ks = sorted(set(da) & set(db))
        if not ks:
            continue
        n01, n10, p = mcnemar([da[k] for k in ks], [db[k] for k in ks])
        m_, lo, hi = boot_ci([db[k] - da[k] for k in ks])
        rows.append([m, str(len(ks)), f"{m_:+.1f} [{lo:+.1f}, {hi:+.1f}]", f"{n01}/{n10}", f"{p:.2g}"])
    return ["Model", "n", "S2M - EB (95% CI)", "wins/losses", "McNemar p"], rows


def ablation_table(model=MODELS[0]):
    ref = run_dir("s2m", model, "final_r1")
    rows = []
    full = per_subset([ref])
    rows.append(["S2M (full)"] + ([fmt(full[s]) for s in SUBSETS] + [fmt(full["avg"])] if full else ["--"] * 6) + [""])
    dref = load(ref) if os.path.isdir(ref) else {}
    for exp, label in ABLATIONS:
        d = run_dir("s2m", model, exp)
        r = per_subset([d])
        if r is None:
            rows.append([label] + ["--"] * 7)
            continue
        dd = load(d)
        ks = sorted(set(dd) & set(dref))
        p = mcnemar([dref[k] for k in ks], [dd[k] for k in ks])[2] if ks else None
        rows.append([label] + [fmt(r[s]) for s in SUBSETS] + [fmt(r["avg"]), "--" if p is None else f"{p:.2g}"])
    return ["Variant"] + [SHORT[s] for s in SUBSETS] + ["Avg", "p vs full"], rows


def robothor_table():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "robothor", "val", "s2m_*"))):  # official val only
        recs = [json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json"))]
        if len(recs) < 1800:
            rows.append([f"S2M ({os.path.basename(d)})", f"{len(recs)}/1800", "-- (incomplete)", "--"])
            continue
        sr = 100 * np.mean([r["success"] for r in recs])
        spl = 100 * np.mean([r["spl"] for r in recs])
        _, lo, hi = boot_ci([r["success"] for r in recs])
        rows.append([f"S2M ({os.path.basename(d)})", str(len(recs)), f"{sr:.1f} [{lo:.1f}, {hi:.1f}]", f"{spl:.1f}"])
    rows += [["SG-Nav (LLaMA-7B) [published, val 1800]", "1800", "47.3", "23.7"],
             ["CoW (OWL-ViT) [published]", "1800", "27.5", "17.2"],
             ["CogNav (GPT-4V) [published, ICCV'25]", "1800", "54.6", "24.3"]]
    return ["Method / run", "episodes", "SR (95% CI)", "SPL"], rows


def write(name, head, rows):
    os.makedirs(OUT, exist_ok=True)
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(OUT, name + ".md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%")
    tex = [r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}", r"\toprule", " & ".join(map(esc, head)) + r" \\",
           r"\midrule"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(tex) + "\n")
    print(f"\n## {name}\n" + "\n".join(md))


if __name__ == "__main__":
    write("main_eb_nav", *main_table())
    write("paired_vs_eb", *paired_rows())
    write("ablations", *ablation_table())
    write("robothor", *robothor_table())
