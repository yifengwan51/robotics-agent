"""Every number, table and figure input of the paper, computed from saved TEST records (no model, no simulator).

  python scripts/paper_assets.py      (env vlm)
    -> paper/latex/sec/numbers.tex   LaTeX macros used in the text
    -> paper/latex/sec/tab_main.tex  main results table
    -> paper/latex/sec/tab_abl.tex   components and sensing table
    -> paper/figures/paper_data.json inputs of scripts/make_paper_figures.py

Statistics. Success is per-episode binary; a configuration run several times enters as its per-episode mean.
95% CIs are episode-level bootstrap intervals. Paired differences use a two-sided sign-flip permutation test on
per-episode differences (exact McNemar for single binary runs), Holm-corrected over the ten variants of tab_abl.
Simulator target state is read here for analysis only; the agents never receive it.
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_nav_failures import classify  # noqa: E402
from stats import SUBSETS, boot_ci, load, mcnemar  # noqa: E402

TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
SEC = os.path.join(ROOT, "paper", "latex", "sec")
FIGDATA = os.path.join(ROOT, "paper", "figures", "paper_data.json")
Q3, Q25 = "Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"
RUN = {
    "eb3": [f"eb_{Q3}-FP8_eb_official_r1"], "eb3_awq": [f"eb_{Q3}-AWQ_eb_official_r1"],
    "s2m3": [f"s2m_{Q3}-AWQ_final_r{i}" for i in (1, 2, 3, 4)],
    "rgb3": [f"s2m_{Q3}-AWQ_rgbonly_floor_r{i}" for i in (1, 2)],
    "eb25": [f"eb_{Q25}-FP8-Dynamic_eb_bounded_rp105_r1"], "eb25_awq": [f"eb_{Q25}-AWQ_eb_bounded_r1"],
    "eb25_awq_raw": [f"eb_{Q25}-AWQ_eb_official_r1"], "s2m25": [f"s2m_{Q25}-AWQ_final_r1"],
    "oracle": ["oracle_geo_none_ceiling"],
}
VARIANTS = [  # (key, table label, run dirs)
    ("fusion", "w/o fusion", [f"s2m_{Q3}-AWQ_abl_no_fusion"]),
    ("scan", "w/o look-around", [f"s2m_{Q3}-AWQ_abl_no_scan"]),
    ("exhaust", "w/o exhaustion", [f"s2m_{Q3}-AWQ_abl_no_exhaust"]),
    ("negative", "w/o neg.\\ evidence", [f"s2m_{Q3}-AWQ_abl_no_negative"]),
    ("tiles", "w/o tiled search", [f"s2m_{Q3}-AWQ_abl_no_tiles"]),
    ("zoom", "w/o crop check", [f"s2m_{Q3}-AWQ_abl_no_zoom"]),
    ("strongest", "strongest first", [f"s2m_{Q3}-AWQ_abl_strongest_first"]),
    ("odom", "odom.\\ noise", [f"s2m_{Q3}-AWQ_odom_noise_r1"]),
    ("rgbglobal", "RGB, global scale", [f"s2m_{Q3}-AWQ_rgbonly_global_r1"]),
    ("rgbfloor", "RGB, floor scale", RUN["rgb3"]),
]
SHORT = ["Base", "Common", "Complex", "Visual", "Long"]
# reported results, each checked against the original paper (method, model, citation, subsets or None, average)
PUBLISHED = [
    ("Prompted MLLMs (EmbodiedBench)", [
        ("EB prompt", "GPT-4o", "yang2025embodiedbench", [55.0, 60.0, 58.3, 60.0, 55.0], 57.7),
        ("EB prompt", "InternVL3-78B", "yang2025embodiedbench", [66.7, 63.3, 61.7, 45.0, 31.7], 53.7),
        ("EB prompt", "Gemini-2.0-flash", "yang2025embodiedbench", [63.3, 65.0, 50.0, 51.7, 13.3], 48.7),
        ("EB prompt", "Claude-3.7-Sonnet", "yang2025embodiedbench", [50.0, 61.7, 50.0, 36.7, 26.7], 45.0),
        ("EB prompt", "Qwen2.5-VL-72B", "yang2025embodiedbench", [46.7, 46.7, 46.7, 26.7, 33.3], 40.0),
        ("EB prompt", "InternVL3-8B", "yang2025embodiedbench", [38.3, 30.0, 40.0, 33.3, 5.0], 29.3),
        ("EB prompt", "Qwen2.5-VL-7B", "yang2025embodiedbench", [20.0, 26.7, 38.3, 16.7, 0.0], 20.3)]),
    ("Training-free agents", [
        ("MOSAIC (module graph)", "Qwen3-VL-8B", "wang2026mosaic", [60.00, 46.67, 45.56, 51.11, 46.67], 50.00),
        ("ESCA (scene graph)", "GPT-4o", "huang2025esca", [66.67, 62.00, 63.33, 55.00, 26.67], 54.67),
        ("EmbodiSkill (skills)", "Qwen3-VL-32B", "ju2026embodiskill", [65.0, 68.0, 73.0, 67.0, 33.0], 61.33),
        ("HODAgent (agent loop)", "Qwen3.6-35B-A3B", "chen2026hodagent", [76.7, 55.0, 68.3, 55.0, 36.7], 58.3),
        ("RoboFoundry (self-evolving)", "Qwen3.8-27B", "liang2026robofoundry", [76.7, 85.0, 73.3, 63.3, 43.3], 68.3),
        ("BrainMem (memory)", "GPT-4o", "ma2026brainmem", [73.3, 71.7, 80.0, 66.6, 63.3], 71.0),
        ("EMHO (harness)", "Qwen3.8-27B", "lee2026emho", [86.7, 76.7, 73.3, 76.7, 63.3], 75.3),
        ("RoboFoundry (self-evolving)", "GPT-6 Astra", "liang2026robofoundry", [90.0, 85.0, 86.7, 73.3, 66.7], 80.3),
        ("Tool library", "Qwen3-VL-8B", "zhou2026enabling", [85.0, 85.0, 77.0, 75.0, 77.0], 80.0),
        ("Tool library", "GPT-4o", "zhou2026enabling", [80.0, 85.0, 85.0, 80.0, 83.0], 83.0)]),
    ("Fine-tuned", [
        ("SVA (Q-model)", "Qwen3.5-4B", "xie2026sva", [54.17, 41.67, 45.83, 41.67, 50.00], 46.67),
        ("MuSix (world models)", "Qwen3-VL-4B", "jang2026musix", [63.2, 63.2, 57.6, 62.4, 43.2], 57.9)]),
]
MACROS = []


def macro(name, value, fmt="{:.1f}"):
    assert name.isalpha(), name
    MACROS.append((name, fmt.format(value) if not isinstance(value, str) else value))


def exists(runs):
    return all(len(glob.glob(os.path.join(TEST, r, "*", "episode_*.json"))) >= 300 for r in runs)


def outcome(runs):
    """Per-episode success averaged over the runs of one configuration."""
    ds = [load(os.path.join(TEST, r)) for r in runs]
    ks = set.intersection(*[set(d) for d in ds])
    return {k: float(np.mean([d[k] for d in ds])) for k in ks}


def records(run):
    return {(os.path.basename(os.path.dirname(f)), os.path.basename(f)): json.load(open(f))
            for f in glob.glob(os.path.join(TEST, run, "*", "episode_*.json"))}


def per_subset(o, ks):
    return [100 * np.mean([o[k] for k in ks if k[0] == s]) for s in SUBSETS]


def perm_test(d, n=1_000_000, seed=0):
    """Two-sided sign-flip permutation p-value for the mean of paired differences d."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs, hits = abs(d.sum()), 0
    for _ in range(n // 20_000):
        s = rng.choice((-1.0, 1.0), size=(20_000, len(d)))
        hits += int((np.abs(s @ d) >= obs - 1e-9).sum())
    return (hits + 1) / (n + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def tex_p(p):
    """p as math-mode content (use inside $...$); values below 1e-3 in scientific notation, below 1e-5 as a bound
    (the resolution of the 10^6-sample permutation test)."""
    if p < 1e-5:
        return "{<}10^{-5}"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 1e-3:
        return f"{p:.3f}"
    e = math.floor(math.log10(p))
    return f"{p / 10 ** e:.1f}{{\\times}}10^{{{e}}}"


def rel_p(p, exact=False):
    """p with its relation for running text: '$p\\nXP$' renders as p=0.22 or p<10^-5."""
    if exact or p >= 1e-5:
        return "{=}" + tex_p(p) if p >= 1e-5 else "{=}" + f"{p / 10 ** math.floor(math.log10(p)):.1f}{{\\times}}10^{{{math.floor(math.log10(p))}}}"
    return "{<}10^{-5}"


def sd_runs(runs):
    v = [100 * np.mean(list(load(os.path.join(TEST, r)).values())) for r in runs]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def exhausted_short(run):
    """Failures that reached a hypothesis within 1 m of the true target and marked it exhausted."""
    n = 0
    for r in records(run).values():
        if r["task_success"] >= 1:
            continue
        t = np.array([r["target"]["x"], r["target"]["z"]])
        n += any(h[3] and np.hypot(h[0] - t[0], h[1] - t[1]) <= 1.0 for e in r["perception_log"] for h in e["hyps"])
    return n


def range_ratios(run):
    """Lifted range / true range of every detection within 1 m of the target (true pose of that step)."""
    out = []
    for r in records(run).values():
        t, tr = np.array([r["target"]["x"], r["target"]["z"]]), r["trajectory"]
        for e in r["perception_log"]:
            if e["step"] >= len(tr):
                continue
            p = np.array([tr[e["step"]]["x"], tr[e["step"]]["z"]])
            rt = np.hypot(*(t - p))
            for d in e["dets"]:
                d = np.array(d[:2])
                if rt >= 0.3 and np.hypot(*(d - t)) <= 1.0:
                    out.append(float(np.hypot(*(d - p)) / rt))
    return out


def candidate_recall(run, radius=0.5):
    hit = {}
    for k, r in records(run).items():
        t = np.array([r["target"]["x"], r["target"]["z"]])
        pts = [p[:2] for e in r["perception_log"] for p in e["dets"] + e["hyps"]]
        hit[k] = any(np.hypot(*(np.array(p) - t)) <= radius for p in pts)
    return hit


def dead_reckoning_error(run):
    """Pose from executed actions vs. simulator pose along the logged trajectories (planar metres)."""
    moves = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
    per_step, final = [], []
    for r in records(run).values():
        tr = r["trajectory"]
        x, z, yaw = tr[0]["x"], tr[0]["z"], tr[0]["yaw"]
        for s in tr[1:]:
            a = s["action"]
            if s["ok"] and a in moves:
                f, g = (0.25 * v for v in moves[a])
                y = math.radians(yaw)
                x += f * math.sin(y) + g * math.cos(y)
                z += f * math.cos(y) - g * math.sin(y)
            elif s["ok"] and a in (4, 5):
                yaw = (yaw + (90 if a == 4 else -90)) % 360
            per_step.append(math.hypot(x - s["x"], z - s["z"]))
        final.append(math.hypot(x - tr[-1]["x"], z - tr[-1]["z"]))
    return float(np.mean(per_step)), float(np.median(final))


def grounding_probe(path):
    recs = [json.loads(line) for line in open(path)]
    out = {}
    for s in SUBSETS[:4]:
        vis = [r for r in recs if r["subset"] == s and r["gt_box"] is not None]  # target instance in view
        out[s] = 100 * np.mean([r["box"] is not None and r.get("iou", 0) >= 0.3 for r in vis])
    return out


def robothor():
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "robothor", "val",
                                                              "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json"))]
    sr = [float(r.get("success", 0) or 0) for r in recs]
    spl = [float(r.get("spl", 0) or 0) for r in recs]
    cats = collections.defaultdict(list)
    for r in recs:
        cats[r["object_type"]].append(float(r.get("success", 0) or 0))
    best = max(cats, key=lambda c: np.mean(cats[c]))
    worst = min(cats, key=lambda c: np.mean(cats[c]))
    return len(recs), boot_ci(sr), boot_ci(spl), (best, 100 * np.mean(cats[best])), (worst, 100 * np.mean(cats[worst]))


def main():
    missing = [k for k, v in RUN.items() if not exists(v)]
    if missing:
        print("WARNING: incomplete runs:", missing)
    O = {k: outcome(v) for k, v in RUN.items() if not k.startswith("eb3") or exists(v)}
    if "eb3" not in O:  # FP8 Qwen3 baseline still running: fall back so the pipeline can be tested
        print("WARNING: eb3 (FP8) incomplete, using AWQ for eb3 in a DRAFT build")
        O["eb3"] = outcome(RUN["eb3_awq"])
    O["eb3_awq"] = outcome(RUN["eb3_awq"])
    ks = sorted(set.intersection(*[set(o) for o in O.values()]))
    assert len(ks) == 300, len(ks)

    # ---------------- main results ----------------
    def sr(o):
        return 100 * np.mean([o[k] for k in ks])

    for key, name in (("eb3", "EBThree"), ("eb3_awq", "EBThreeAWQ"), ("s2m3", "SThree"), ("rgb3", "RGBThree"),
                      ("eb25", "EBTwo"), ("eb25_awq", "EBTwoAWQ"), ("eb25_awq_raw", "EBTwoAWQraw"),
                      ("s2m25", "STwo"), ("oracle", "Oracle")):
        subs = per_subset(O[key], ks)
        m, lo, hi = boot_ci([O[key][k] for k in ks])
        macro(name, m); macro(name + "Lo", lo); macro(name + "Hi", hi)
        for s, v in zip(SHORT, subs):
            macro(name + s, v)
    for name, a, b in (("GainThree", "eb3", "s2m3"), ("GainTwo", "eb25", "s2m25"), ("GainRGB", "eb3", "rgb3"),
                       ("GainThreeAWQ", "eb3_awq", "s2m3"), ("GainTwoAWQ", "eb25_awq", "s2m25"),
                       ("QuantThree", "eb3_awq", "eb3"), ("QuantTwo", "eb25_awq", "eb25"),
                       ("GapRGB", "s2m3", "rgb3"), ("GapOracle", "s2m3", "oracle")):
        d = [O[b][k] - O[a][k] for k in ks]
        m, lo, hi = boot_ci(d)
        macro(name, m); macro(name + "Lo", lo); macro(name + "Hi", hi)
        macro(name + "Abs", abs(m))
        p = perm_test(d)
        macro(name + "P", rel_p(p))
        for s, sn in zip(SUBSETS, SHORT):  # signed: use as $\\nGainThreeBase$
            macro(name + sn, 100 * np.mean([O[b][k] - O[a][k] for k in ks if k[0] == s]), "{:+.1f}")
    # single binary runs against the FP8 baselines: discordant pairs and exact McNemar
    for name, base, runs in (("Three", "eb3", RUN["s2m3"]), ("Two", "eb25", RUN["s2m25"]), ("RGB", "eb3", RUN["rgb3"])):
        worst = 0.0
        for r in runs:
            d = load(os.path.join(TEST, r))
            _, _, p = mcnemar([O[base][k] for k in ks], [d[k] for k in ks])
            worst = max(worst, p)
        macro(f"McN{name}Worst", rel_p(worst, exact=True))
    for name, a, b in (("Two", "eb25", "s2m25"), ("Three", "eb3", "s2m3"), ("Global", "eb3", "rgbglobal")):
        if b == "rgbglobal":
            ob = outcome(VARIANTS[8][2])
        else:
            ob = load(os.path.join(TEST, RUN[b][0]))
        n01, n10, p = mcnemar([O[a][k] for k in ks], [ob[k] for k in ks])
        macro(f"Disc{name}Wins", n01, "{:d}"); macro(f"Disc{name}Losses", n10, "{:d}"); macro(f"Disc{name}P", rel_p(p, exact=True))
    og = outcome(VARIANTS[8][2])
    macro("RGBGlobal", 100 * np.mean([og[k] for k in ks]))
    macro("GainGlobal", 100 * np.mean([og[k] - O["eb3"][k] for k in ks]), "{:+.1f}")
    # Visual: S2M against the 8-bit EB prompt, and where 8-bit weights help the EB prompt
    vk = [k for k in ks if k[0] == "visual_appearance"]
    dv = [O["s2m3"][k] - O["eb3"][k] for k in vk]
    m, lo, hi = boot_ci(dv)
    macro("VisDiffLo", lo, "{:+.1f}"); macro("VisDiffHi", hi, "{:+.1f}"); macro("VisDiffP", rel_p(perm_test(dv)))
    qn = [100 * np.mean([O["eb3"][k] - O["eb3_awq"][k] for k in ks if k[0] == s]) for s in SUBSETS[:3]]
    macro("QuantThreeNamedMax", max(qn), "{:+.1f}")
    for q in ("eb3", "eb25"):
        a, b = (O["eb3_awq"], O["eb3"]) if q == "eb3" else (O["eb25_awq"], O["eb25"])
        n01, n10, p = mcnemar([a[k] for k in ks], [b[k] for k in ks])
        tag = "Three" if q == "eb3" else "Two"
        macro(f"Quant{tag}Wins", n01, "{:d}"); macro(f"Quant{tag}Losses", n10, "{:d}")
    macro("SThreeSD", sd_runs(RUN["s2m3"])); macro("RGBThreeSD", sd_runs(RUN["rgb3"]))
    vals = [100 * np.mean(list(load(os.path.join(TEST, r)).values())) for r in RUN["s2m3"]]
    macro("SThreeMin", min(vals)); macro("SThreeMax", max(vals))
    macro("ScaleDrop", sr(O["s2m3"]) - sr(O["s2m25"])); macro("ScaleDropEB", sr(O["eb3"]) - sr(O["eb25"]))
    macro("OracleGapLong", per_subset(O["oracle"], ks)[4] - per_subset(O["s2m3"], ks)[4])

    # steps on the episodes that the FP8 baseline and every S2M run solve
    R_eb = records(RUN["eb3"][0] if exists(RUN["eb3"]) else RUN["eb3_awq"][0])
    R_s = [records(r) for r in RUN["s2m3"]]
    both = [k for k in ks if R_eb[k]["task_success"] >= 1 and all(R[k]["task_success"] >= 1 for R in R_s)]
    a = np.array([np.mean([R[k]["num_steps"] for R in R_s]) for k in both])
    b = np.array([R_eb[k]["num_steps"] for k in both])
    f, g = int((a < b - 1e-9).sum()), int((a > b + 1e-9).sum())
    p = min(1.0, 2 * sum(math.comb(f + g, i) for i in range(min(f, g) + 1)) / 2 ** (f + g))
    macro("StepN", len(both), "{:d}"); macro("StepS", a.mean()); macro("StepEB", b.mean())
    macro("StepRed", 100 * (1 - a.mean() / b.mean()), "{:.0f}"); macro("StepFewer", f, "{:d}")
    macro("StepMore", g, "{:d}"); macro("StepP", rel_p(p, exact=True))

    # compute: tokens per episode (S2M run 4 logs tokens) and wall-clock
    R4 = records(RUN["s2m3"][3])
    macro("TokS", np.mean([r["vlm_prompt_tokens"] for r in R4.values()]) / 1000)
    macro("TokSc", np.mean([r["vlm_completion_tokens"] for r in R4.values()]), "{:.0f}")
    macro("CallsS", np.mean([r["vlm_calls"] for r in R4.values()]))
    macro("TokEB", np.mean([r["model_calls"]["prompt_tokens"] for r in R_eb.values()]) / 1000)
    macro("TokEBc", np.mean([r["model_calls"]["completion_tokens"] for r in R_eb.values()]) / 1000)
    macro("CallsEB", np.mean([r["model_calls"]["calls"] for r in R_eb.values()]))
    ne, nc = sum(r.get("json_errors", 0) for r in R_eb.values()), sum(r["model_calls"]["calls"] for r in R_eb.values())
    macro("EBThreeParse", f"all {nc:,d} of its replies parse" if ne == 0 else
          (f"{ne} of its {nc:,d} replies " + ("cannot be parsed and is replaced" if ne == 1 else "cannot be parsed and are replaced")
           + " by a random action, as in the benchmark"))

    # outcome categories (figure) and the counts used in the text
    cats = ["success", "never_seen", "lost_after_seen", "near_miss", "approach_fail", "blocked", "oscillation"]
    taxo = {}
    for key, runs in (("eb3", RUN["eb3"] if exists(RUN["eb3"]) else RUN["eb3_awq"]), ("s2m3", RUN["s2m3"]),
                      ("rgb3", RUN["rgb3"]), ("eb25", RUN["eb25"]), ("s2m25", RUN["s2m25"])):
        cs = [collections.Counter(classify(r) for r in records(run).values()) for run in runs]
        taxo[key] = {k: float(np.mean([c.get(k, 0) for c in cs])) for k in cats}  # mean over the runs
    for key, name in (("eb3", "EBThree"), ("s2m3", "SThree")):
        t = taxo[key]
        fmt = "{:.0f}" if key == "eb3" else "{:.1f}"
        macro(f"Fail{name}", 300 - t["success"], fmt)
        for c, n in (("never_seen", "Never"), ("lost_after_seen", "Lost"), ("near_miss", "Near"), ("blocked", "Blocked")):
            macro(f"{n}{name}", t[c], fmt)
    lh_eb = sum(1 for k, r in R_eb.items() if k[0] == "long_horizon" and classify(r) == "never_seen")
    macro("NeverLongEBThree", lh_eb, "{:d}")

    # recognition: candidate recall of S2M (4 runs) and Visual failures without a candidate
    rec = [candidate_recall(r) for r in RUN["s2m3"]]
    other = [100 * np.mean([h[k] for k in ks if k[0] == s]) for h in rec for s in SUBSETS if s != "visual_appearance"]
    vis = [100 * np.mean([h[k] for k in ks if k[0] == "visual_appearance"]) for h in rec]
    macro("RecallOtherLo", min(other), "{:.0f}"); macro("RecallOtherHi", max(other), "{:.0f}")
    macro("RecallVisLo", min(vis), "{:.0f}"); macro("RecallVisHi", max(vis), "{:.0f}")
    vf = vnc = 0
    for h, R in zip(rec, R_s):
        for k in ks:
            if k[0] == "visual_appearance" and R[k]["task_success"] < 1:
                vf += 1; vnc += not h[k]
    macro("VisFail", vf, "{:d}"); macro("VisFailNoCand", vnc, "{:d}")
    vis_sr = [100 * np.mean([R[k]["task_success"] >= 1 for k in ks if k[0] == "visual_appearance"]) for R in R_s]
    macro("VisSRLo", min(vis_sr)); macro("VisSRHi", max(vis_sr))

    # single-view probe (start view and the three other 90 degree headings)
    for model, name in ((f"{Q3}-AWQ", "Three"), (f"{Q25}-AWQ", "Two")):
        g = grounding_probe(os.path.join(ROOT, "runs", "grounding", f"{model}_auto_test.jsonl"))
        for s, sn in zip(SUBSETS[:4], SHORT[:4]):
            macro(f"Probe{name}{sn}", g[s])

    # sensing: pose, depth bias, exhausted-short failures, floor probe
    me, fe = dead_reckoning_error(RUN["s2m3"][0])
    macro("DRmean", me, "{:.2f}"); macro("DRfinal", fe, "{:.2f}")
    rr = {"depth": sum((range_ratios(r) for r in RUN["s2m3"]), []), "global": range_ratios(VARIANTS[8][2][0]),
          "floor": sum((range_ratios(r) for r in RUN["rgb3"]), [])}  # pooled over the runs of each configuration
    for k, n in (("depth", "Sensor"), ("global", "Global"), ("floor", "Floor")):
        macro(f"Range{n}", float(np.median(rr[k])), "{:.2f}")
    macro("ShortGlobal", exhausted_short(VARIANTS[8][2][0]), "{:d}")
    macro("ShortFloor", np.mean([exhausted_short(r) for r in RUN["rgb3"]]), "{:.1f}")
    fp = json.load(open(os.path.join(ROOT, "paper", "tables", "floor_probe_eb_navigation.json")))
    macro("ProbeViews", fp["frames"], "{:d}")
    macro("BlockGlobal", 100 * fp["global"]["floor_as_obstacle"]); macro("BlockFloor", 100 * fp["floor"]["floor_as_obstacle"])
    macro("AbsRelGlobal", fp["global"]["absrel"], "{:.3f}"); macro("AbsRelFloor", fp["floor"]["absrel"], "{:.3f}")

    macro("ShortSensorMax", max(exhausted_short(r) for r in RUN["s2m3"]), "{:d}")
    # Visual share of the gap to the oracle
    gap = [o - m for o, m in zip(per_subset(O["oracle"], ks), per_subset(O["s2m3"], ks))]
    macro("VisGapShare", 100 * gap[3] / sum(gap), "{:.0f}")
    # EB-prompt output validity, Qwen2.5 (captured raw replies): plans at the 20-action cap, action id = action name
    names = ["move forward", "move backward", "move rightward", "move leftward", "rotate to the right",
             "rotate to the left", "tilt the camera upward", "tilt the camera downward"]

    def canon(n):
        n = n.lower()
        for i, k in enumerate(names):
            if k in n:
                return i
        return {"move right": 2, "move left": 3}.get(next((k for k in ("move right", "move left") if k in n), ""), None)
    for d, tag in (("qwen25_eb_bounded_test", "AWQ"), ("qwen25_eb_fp8_test", "FP")):
        cap = calls = 0; first = [0, 0]; rest = [0, 0]
        for f in glob.glob(os.path.join(ROOT, "runs", "diag", d, "responses_*.jsonl")):
            for line in open(f):
                calls += 1
                try:
                    plan = json.loads(json.loads(line)["raw"]).get("executable_plan", [])
                except Exception:
                    continue
                cap += len(plan) >= 20
                for j, x in enumerate(plan):
                    c = canon(x.get("action_name", ""))
                    if c is not None:
                        bucket = first if j == 0 else rest
                        bucket[0] += 1; bucket[1] += int(c == x.get("action_id"))
        macro(f"Cap{tag}", 100 * cap / calls); macro(f"IdFirst{tag}", 100 * first[1] / first[0])
        macro(f"IdLater{tag}", 100 * rest[1] / rest[0])

    # RoboTHOR
    n, (srm, srl, srh), (splm, spll, splh), best, worst = robothor()
    macro("RoboN", n, "{:d}"); macro("RoboSR", srm); macro("RoboSRLo", srl); macro("RoboSRHi", srh)
    macro("RoboSPL", splm); macro("RoboSPLLo", spll); macro("RoboSPLHi", splh)
    macro("RoboBest", best[1]); macro("RoboWorst", worst[1])
    macro("RoboErr", sum(1 for f in glob.glob(os.path.join(ROOT, "runs", "robothor", "val", "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json"))
                         if json.load(open(f)).get("error")), "{:d}")

    # ---------------- components and sensing (table) ----------------
    ref = O["s2m3"]
    rows, ps = [], []
    for key, label, runs in VARIANTS:
        o = outcome(runs)
        d = [o[k] - ref[k] for k in ks]
        m, lo, hi = boot_ci(d)
        ps.append(perm_test(d))
        rows.append((label, per_subset(o, ks), 100 * np.mean([o[k] for k in ks]), m, lo, hi, key))
    adj = holm(ps)
    for (label, subs, avg, m, lo, hi, key), q in zip(rows, adj):
        nm = "".join(w.capitalize() for w in key.split())
        macro(f"Abl{nm.capitalize()}", m); macro(f"Abl{nm.capitalize()}Abs", abs(m))
        macro(f"Abl{nm.capitalize()}Lo", lo); macro(f"Abl{nm.capitalize()}Hi", hi)
        macro(f"Abl{nm.capitalize()}P", rel_p(q))
        dsub = [v - r for v, r in zip(subs, per_subset(ref, ks))]
        for s, v in zip(SHORT, dsub):  # signed: use as $\\nAblScanLong$
            macro(f"Abl{nm.capitalize()}{s}", v, "{:+.1f}")
        macro(f"Abl{nm.capitalize()}OtherMin", min(dsub[:4]), "{:+.1f}")
        macro(f"Abl{nm.capitalize()}OtherMax", max(dsub[:4]), "{:+.1f}")
    with open(os.path.join(SEC, "tab_abl.tex"), "w") as f:
        f.write("% generated by scripts/paper_assets.py -- do not edit\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{6pt}\n")
        f.write("\\caption{Components and sensing on EB-Navigation test (Qwen3-VL-4B). Each row changes one element of "
                "\\ours. $\\Delta$: paired difference in average success from the full model, with its 95\\% CI; "
                "$p$: sign-flip permutation test, Holm-corrected over the ten rows. Bold: significant.}\n\\label{tab:abl}\n")
        f.write("\\begin{tabular}{@{}lrrrrrrrlr@{}}\n\\toprule\n"
                "Variant & Base & Common & Complex & Visual & Long & Avg. & \\multicolumn{2}{c}{$\\Delta$ (95\\% CI)} & $p$\\\\\n\\midrule\n")
        subs = per_subset(ref, ks)
        f.write(f"\\ours (full) & {' & '.join(f'{v:.1f}' for v in subs)} & {np.mean([ref[k] for k in ks]) * 100:.1f} & "
                "\\multicolumn{2}{c}{--} & --\\\\\n")
        labels = {"odom": "odometry noise (5\\%, 2$^\\circ$)", "rgbglobal": "RGB only, global scale",
                  "rgbfloor": "RGB only, floor scale"}
        for i, ((label, subs, avg, m, lo, hi, key), q) in enumerate(zip(rows, adj)):
            if i in (0, 7):
                f.write("\\midrule\n\\multicolumn{10}{@{}l}{\\textit{" + ("Components" if i == 0 else "Sensing") + "}}\\\\\n")
            d = f"${m:+.1f}$"
            if q < 0.05:
                d = f"$\\mathbf{{{m:+.1f}}}$"
            f.write(f"{labels.get(key, label)} & {' & '.join(f'{v:.1f}' for v in subs)} & {avg:.1f} & {d} & "
                    f"{{\\footnotesize[${lo:+.1f}$, ${hi:+.1f}$]}} & ${tex_p(q)}$\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---------------- main table ----------------
    def ours(key, method, model, sd=None, block=()):
        """One row; in each column the best entry of its model block is bold."""
        o = O[key]
        cols = per_subset(o, ks) + [100 * np.mean([o[k] for k in ks])]
        best = [max(per_subset(O[b], ks)[i] if i < 5 else 100 * np.mean([O[b][k] for k in ks]) for b in block)
                for i in range(6)] if block else [None] * 6
        m, lo, hi = boot_ci([o[k] for k in ks])
        cells = [(f"\\textbf{{{v:.1f}}}" if best[i] is not None and round(v, 1) >= round(best[i], 1) else f"{v:.1f}")
                 for i, v in enumerate(cols)]
        if sd is not None:
            cells[5] += f"\\,{{\\scriptsize$\\pm${sd:.1f}}}"
        return f"{method} & {model} & {' & '.join(cells)} & [{lo:.1f}, {hi:.1f}]\\\\\n"

    with open(os.path.join(SEC, "tab_main.tex"), "w") as f:
        f.write("% generated by scripts/paper_assets.py -- do not edit\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{5pt}\n")
        f.write("\\caption{Success rate (\\%) on EB-Navigation test. Top: reported results of prompted MLLMs, "
                "training-free agents, and fine-tuned agents. Bottom: our runs on one 8\\,GB GPU; the EB prompt is "
                "the benchmark's agent with the same model; \\ours runs the model at 4-bit. "
                "$\\pm$: standard deviation over repeated test runs. "
                "CI: episode-level bootstrap. Bold: best in each column among our runs of the same model.}\n\\label{tab:main}\n")
        f.write("\\begin{tabular}{@{}llrrrrrrc@{}}\n\\toprule\n"
                "Method & Model & Base & Common & Complex & Visual & Long & Avg. & 95\\% CI\\\\\n\\midrule\n")
        for group, rows in PUBLISHED:
            f.write(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{group}}}}}\\\\\n")
            for name, size, cite, subs, avg in rows:
                cells = " & ".join(f"{v:.1f}" for v in subs) if subs else " & ".join(["--"] * 5)
                f.write(f"{name}~\\cite{{{cite}}} & {size} & {cells} & {avg:.1f} & --\\\\\n")
        f.write("\\midrule\n\\multicolumn{9}{@{}l}{\\textit{Ours (one 8\\,GB GPU)}}\\\\\n")
        q3, q25 = ("eb3", "s2m3", "rgb3"), ("eb25", "s2m25")
        f.write(ours("eb3", "EB prompt", "Qwen3-VL-4B", block=q3))
        f.write(ours("s2m3", "\\ours", "Qwen3-VL-4B", sd_runs(RUN["s2m3"]), block=q3))
        f.write(ours("rgb3", "\\ours, RGB only$^\\dagger$", "Qwen3-VL-4B", sd_runs(RUN["rgb3"]), block=q3))
        f.write("\\midrule\n")
        f.write(ours("eb25", "EB prompt", "Qwen2.5-VL-3B", block=q25))
        f.write(ours("s2m25", "\\ours", "Qwen2.5-VL-3B", block=q25))
        f.write("\\midrule\n")
        f.write(ours("oracle", "Oracle target$^\\ddagger$", "--"))
        f.write("\\bottomrule\n\\end{tabular}\n\n\\vspace{0.2em}\\footnotesize "
                "$^\\dagger$No depth sensor and no simulator pose (\\cref{sec:sensing}). "
                "$^\\ddagger$Geometric planner given the true target position: an upper bound.\n\\end{table*}\n")

    # ---------------- figure data ----------------
    budget = {}
    for key, runs in (("eb3", RUN["eb3"] if exists(RUN["eb3"]) else RUN["eb3_awq"]), ("s2m3", RUN["s2m3"]),
                      ("rgb3", RUN["rgb3"]), ("oracle", RUN["oracle"])):
        curves = []
        for r in runs:
            R = records(r)
            steps = [R[k]["num_steps"] if R[k]["task_success"] >= 1 else 99 for k in ks]
            curves.append([100 * np.mean([s <= t for s in steps]) for t in range(21)])
        budget[key] = np.mean(curves, axis=0).tolist()
    for key, n in (("eb3", "EB"), ("s2m3", "S"), ("rgb3", "RGB"), ("oracle", "Oracle")):
        macro(f"Bud{n}Ten", budget[key][10]); macro(f"Bud{n}Fifteen", budget[key][15])
    # every qualitative statement of the text, checked against the numbers it rests on
    M = dict(MACROS)
    num = lambda k: float(M[k].replace("{=}", "").replace("+", "").replace(",", "").split("{")[0])
    claims = {
        "S2M gains on Base, Common, Complex and Long (Qwen3)": all(num(f"GainThree{x}") > 0 for x in SHORT if x != "Visual"),
        "Visual: S2M within 3 pp of the 8-bit EB prompt, not significant": abs(num("GainThreeVisual")) < 3
            and "times" not in M["VisDiffP"] and float(M["VisDiffP"].replace("{=}", "")) >= 0.05,
        "S2M gains on every subset (Qwen2.5)": all(num(f"GainTwo{x}") > 0 for x in SHORT),
        "RGB-only gains on Base, Common, Complex and Long": all(num(f"GainRGB{x}") > 0 for x in SHORT if x != "Visual"),
        "largest gain on Long": max(SHORT, key=lambda x: num(f"GainThree{x}")) == "Long",
        "fewer never-seen": num("NeverSThree") < num("NeverEBThree"),
        "fewer blocked": num("BlockedSThree") < num("BlockedEBThree"),
        "fewer lost after seen": num("LostSThree") < num("LostEBThree"),
        "fewer near misses": num("NearSThree") < num("NearEBThree"),
        "fewer prompt tokens": num("TokS") < num("TokEB"),
        "more VLM calls": num("CallsS") > num("CallsEB"),
        "ahead at step budget 10 and 15": num("BudSTen") > num("BudEBTen") and num("BudSFifteen") > num("BudEBFifteen"),
        "EB prompt loses more from 4B to 3B": num("ScaleDropEB") > num("ScaleDrop"),
        "fusion is the only significant component": [k for k, _, _ in VARIANTS[:7] if adj[[v[0] for v in VARIANTS].index(k)] < 0.05] == ["fusion"],
        "RGB global is not significant vs EB": float(M["DiscGlobalP"].replace("{=}", "").split("{")[0]) >= 0.05 if "times" not in M["DiscGlobalP"] else False,
        "Visual is the weakest S2M subset": min(SHORT, key=lambda x: num(f"SThree{x}")) == "Visual",
        "S2M above every prompted MLLM in EmbodiedBench": all(num("SThree") > r[4] for r in PUBLISHED[0][1]),
        "S2M above every published agent but the tool library and RoboFoundry (GPT-6 Astra)": all(
            num("SThree") > r[4] for g in PUBLISHED[1:] for r in g[1]
            if r[0] != "Tool library" and r[1] != "GPT-6 Astra"),
        "S2M below both tool-library agents and RoboFoundry (GPT-6 Astra) (stated in the text)":
            all(num("SThree") < v for v in (83.0, 80.0, 80.3)),
        "RoboTHOR: S2M SR above every zero-shot method but ForesightNav and STRIVE": all(num("RoboSR") > v for v in
            (26.7, 35.0, 38.1, 41.2, 42.3, 44.1, 47.5, 48.0, 54.6, 57.5)) and num("RoboSR") < min(62.5, 68.1),
        "RoboTHOR: S2M SPL above every method but STRIVE": all(num("RoboSPL") > v for v in
            (16.9, 21.9, 22.2, 22.5, 23.0, 23.3, 24.0, 24.2, 24.3, 26.8, 27.4, 32.9)) and num("RoboSPL") < 36.3,
        "stop-short failures: 0 with sensor, fewer with floor than global": num("ShortSensorMax") == 0 and num("ShortFloor") < num("ShortGlobal"),
        "floor scale reduces blocked floor and range bias": num("BlockFloor") < num("BlockGlobal") and abs(1 - num("RangeFloor")) < abs(1 - num("RangeGlobal")),
    }
    for k, ok in claims.items():
        print(("  ok   " if ok else "  FAIL ") + k)
    with open(os.path.join(SEC, "numbers.tex"), "w") as f:
        f.write("% generated by scripts/paper_assets.py from the TEST records -- do not edit\n")
        for name, v in MACROS:
            f.write(f"\\newcommand{{\\n{name}}}{{{v}}}\n")

    cats = collections.defaultdict(lambda: [[], []])
    for fn in glob.glob(os.path.join(ROOT, "runs", "robothor", "val", "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json")):
        r = json.load(open(fn))
        cats[r["object_type"]][0].append(float(r.get("success", 0) or 0))
        cats[r["object_type"]][1].append(float(r.get("spl", 0) or 0))
    data = {"subsets": SHORT, "budget": budget, "taxonomy": taxo,
            "published": [[g, name, model, avg] for g, rows in PUBLISHED for name, model, _, _, avg in rows],
            "robothor_cats": {c: [100 * np.mean(v[0]), 100 * np.mean(v[1]), len(v[0])] for c, v in cats.items()},
            "robothor_sr": srm,
            "bars": {k: per_subset(O[k], ks) + [100 * np.mean([O[k][x] for x in ks])]
                     for k in ("eb3", "s2m3", "rgb3", "eb25", "s2m25", "oracle")},
            "range_ratio": rr, "draft": "eb3" in missing}
    os.makedirs(os.path.dirname(FIGDATA), exist_ok=True)
    json.dump(data, open(FIGDATA, "w"))
    print(f"wrote {len(MACROS)} macros, tab_main.tex, tab_abl.tex, {FIGDATA}" + ("  [DRAFT: eb3 FP8 missing]" if missing else ""))


if __name__ == "__main__":
    main()
