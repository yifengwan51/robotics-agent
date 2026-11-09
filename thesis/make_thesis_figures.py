"""Figures used only by the thesis, drawn from the generated numbers and the saved test records.

  ~/miniforge3/envs/embnav/bin/python thesis/make_thesis_figures.py   -> thesis/figures/th_*.pdf
Palette: the paper's (S2M orange, RGB-only light orange, EB prompt grey) plus blue for the generic modular agent;
the light and grey series carry hatching and direct labels so identity never rests on colour alone.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = os.path.join(ROOT, "thesis")
FIG, GEN = os.path.join(TH, "figures"), os.path.join(TH, "chapters", "generated")
TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_paper_figures import check  # noqa: E402

S2M, RGBO, EB, MOD, ORC = "#C55A11", "#F4B183", "#8C8C8C", "#2E75B6", "#404040"
INK, INK2, GRID = "#0b0b0b", "#595959", "#E7E6E6"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2})
def read_macros(path):
    """\\newcommand{\\nName}{value} with nested braces in value (e.g. {=}8.1{\\times}10^{-5})."""
    out, t = {}, open(path).read()
    for m in re.finditer(r"\\newcommand\{\\n(\w+)\}\{", t):
        i, d = m.end(), 1
        j = i
        while d:
            d += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        out[m.group(1)] = t[i:j - 1]
    return out


def p_value(tex):
    """Numeric value of a p macro such as {=}0.57, {<}10^{-5}, {=}8.1{\\times}10^{-5}."""
    s = tex.replace("{", "").replace("}", "").lstrip("=<")
    m = re.match(r"^(?:([\d.]+)\\times)?10\^-?(\d+)$", s.replace("{", ""))
    if m:
        return float(m.group(1) or 1) * 10 ** -int(m.group(2))
    return float(s)


N = read_macros(os.path.join(GEN, "numbers.tex"))
num = lambda k: float(N[k])
SUBS = ["Base", "Common", "Complex", "Visual", "Long"]


def style(ax, ygrid=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def save(fig, name):
    check(fig, name)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)


# 1. success per subset: the ladder from the prompted agent to the oracle
def subsets():
    series = [("EB prompt", "EBThree", EB, ""), ("Modular, single view", "ModThree", MOD, ""),
              ("S2M", "SThree", S2M, ""), ("S2M, RGB only", "RGBThree", RGBO, "////"), ("Oracle target", "Oracle", ORC, "")]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    x, w = np.arange(len(SUBS)), 0.16
    for i, (lab, key, c, h) in enumerate(series):
        v = [num(key + s) for s in SUBS]
        ax.bar(x + (i - 2) * w, v, w * 0.92, color=c, label=lab, hatch=h, edgecolor="white" if not h else S2M, lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBS)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.12)
    save(fig, "th_subsets")


# 2. single-view probe: recall on views with the target against boxes on views without it
def probe():
    t = open(os.path.join(GEN, "supp_tab_probe.tex")).read()
    rows = {m.group(1): [float(v) for v in m.group(2).split("&")] for m in
            re.finditer(r"^(Qwen[\w.-]+) & ([\d. &]+)\\\\", t, re.M)}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for ax, (title, sl) in zip(axes, (("Box on the visible target", slice(0, 4)), ("Box on a view without the target", slice(4, 8)))):
        x = np.arange(4)
        for i, (m, c, h) in enumerate((("Qwen3-VL-4B", S2M, ""), ("Qwen2.5-VL-3B", RGBO, "////"))):
            v = rows[m][sl]
            b = ax.bar(x + (i - 0.5) * 0.36, v, 0.33, color=c, label=m, hatch=h, edgecolor="white" if not h else S2M, lw=0.4)
            for xi, vi in zip(x + (i - 0.5) * 0.36, v):
                ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(SUBS[:4])
        ax.set_title(title, color=INK)
        ax.set_ylim(0, 95)
        style(ax)
    axes[0].set_ylabel("Views (%)")
    axes[1].legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.12, wspace=0.06)
    save(fig, "th_probe")


# 3. outcomes of every agent: stacked bars over the 300 test episodes
def outcomes():
    t = open(os.path.join(GEN, "supp_tab_taxonomy.tex")).read()
    rows = [(m.group(1), [float(v) for v in m.group(2).split("&")]) for m in
            re.finditer(r"^([a-z ]+) & ([\d. &]+)\\\\", t, re.M)]
    agents = ["EB prompt\n(Qwen3)", "S2M\n(Qwen3)", "S2M, RGB\n(Qwen3)", "EB prompt\n(Qwen2.5)", "S2M\n(Qwen2.5)"]
    cols = {"success": S2M, "never seen": "#7F7F7F", "lost after seen": "#2E75B6", "near miss": "#9DC3E6",
            "blocked": "#404040", "oscillation": "#BFBFBF", "approach failure": "#DEEBF7"}
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    left = np.zeros(len(agents))
    y = np.arange(len(agents))[::-1]
    for name, v in rows:
        v = np.array(v)
        ax.barh(y, v, 0.62, left=left, color=cols[name], label=name, edgecolor="white", lw=0.8)
        for yi, li, vi in zip(y, left, v):
            if vi >= 18:
                ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if name in ("success", "lost after seen", "blocked") else INK)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(agents, fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Episodes (of 300)")
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.25), frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.16)
    save(fig, "th_outcomes")


# 4. ablation forest plot: paired difference from the full agent with 95% CI
def forest():
    items = [("w/o fusion", "Fusion"), ("w/o look-around", "Scan"), ("w/o exhaustion", "Exhaust"),
             ("w/o neg. evidence", "Negative"), ("w/o tiled search", "Tiles"), ("w/o crop check", "Zoom"),
             ("strongest first", "Strongest"), ("odometry noise", "Odom"), ("RGB only, global scale", "Rgbglobal"),
             ("RGB only, floor scale", "Rgbfloor")]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y = np.arange(len(items))[::-1]
    for yi, (lab, k) in zip(y, items):
        m, lo, hi = num(f"Abl{k}"), num(f"Abl{k}Lo"), num(f"Abl{k}Hi")
        sig = p_value(N[f"Abl{k}P"]) < 0.05
        c = S2M if sig else INK2
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot([m], [yi], "o", color=c, ms=5, mec="white", mew=0.8)
        ax.text(max(hi, 0) + 1.0, yi, f"{m:+.1f}" + (" *" if sig else ""), va="center", fontsize=7.5, color=c)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel("Paired difference from the full agent (pp), 95% CI")
    ax.set_xlim(-36, 14)
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.26, right=0.99, top=0.97, bottom=0.16)
    save(fig, "th_forest")


# 5. one episode: camera frames with the VLM's box, and the evidence of every hypothesis over the steps
def episode():
    run = "s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1"
    ep = json.load(open(os.path.join(TEST, run, "long_horizon", "episode_42.json")))
    rep = os.path.join(ROOT, "paper", "figures", "scene_replay", "s2m_qwen3_r1_long_42")
    man = json.load(open(os.path.join(rep, "manifest.json")))
    frames = {f["step"]: f["rgb"] for f in man["frames"]}
    t = ep["target"]
    near = lambda x, z: np.hypot(x - t["x"], z - t["z"]) < 0.5
    log = {e["step"]: e for e in ep["perception_log"]}
    show = [0, 2, 7, 13]
    fig = plt.figure(figsize=(6.3, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.05)
    for i, s in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(Image.open(os.path.join(rep, frames[s])))
        e = log[s]
        if e.get("box") is not None:
            x0, y0, x1, y1 = e["box"]
            on = e["dets"] and near(*e["dets"][0][:2])
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.4,
                                       ec=S2M if on else "#2E75B6", ls="-" if on else (0, (3, 1.5))))
        ax.set_title(f"step {s}", fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    ax = fig.add_subplot(gs[1, :])
    steps = [e["step"] for e in ep["perception_log"]]
    wt = [next((h[2] for h in e["hyps"] if near(h[0], h[1])), np.nan) for e in ep["perception_log"]]
    ax.plot(steps, wt, "-o", color=S2M, ms=3.5, lw=1.6, label="target")
    look = {}
    for e in ep["perception_log"]:
        for hx, hz, w, _ in e["hyps"]:
            if not near(hx, hz):
                k = next((q for q in look if np.hypot(q[0] - hx, q[1] - hz) < 0.25), (hx, hz))
                look.setdefault(k, []).append((e["step"], w))
    for j, tr in enumerate(look.values()):
        ax.plot([a for a, _ in tr], [b for _, b in tr], "--s", color="#2E75B6", ms=3, lw=1.2,
                label="look-alikes" if j == 0 else None)
    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 2)))
    ax.text(-0.2, 0.95, r"commit $\tau_c$", fontsize=7.5, color=INK2, ha="left", va="bottom")
    ax.axhline(-1.0, color=INK2, lw=0.6, ls=":")
    ax.text(-0.2, -0.92, "removal", fontsize=7.5, color=INK2, ha="left", va="bottom")
    for s in show:
        ax.axvline(s, color=GRID, lw=3, zorder=0)
    ax.set_xlim(-0.3, 13.6)
    ax.set_ylim(-1.3, 8.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Evidence $w$")
    style(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.95, bottom=0.11)
    save(fig, "th_episode")


# 6. RoboTHOR: how episodes end
def robothor():
    R = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "robothor", "val",
                                                          "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json"))]
    ok = [r for r in R if not r.get("error")]
    stop = [r for r in ok if r["modes"].endswith("S")]
    parts = [("success (Stop at the target)", sum(r["success"] >= 1 for r in stop), S2M),
             ("Stop away from the target", sum(r["success"] < 1 for r in stop), "#2E75B6"),
             ("step limit (500)", sum(r["steps"] >= 500 for r in ok), "#BFBFBF"),
             ("start pose rejected", len(R) - len(ok), "#404040")]
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    left = 0
    for lab, v, c in parts:
        ax.barh([0], [v], 0.55, left=left, color=c, edgecolor="white", lw=1.0, label=f"{lab}: {v}")
        if v > 60:
            ax.text(left + v / 2, 0, f"{v}", ha="center", va="center", fontsize=8, color="white" if c != "#BFBFBF" else INK)
        left += v
    ax.set_xlim(0, len(R))
    ax.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    ax.set_xlabel(f"RoboTHOR validation episodes (of {len(R)})")
    style(ax, ygrid=False)
    ax.spines["left"].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               frameon=False, handlelength=1.0)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.62, bottom=0.27)
    save(fig, "th_robothor_outcomes")
    fs = [r for r in stop if r["success"] < 1]
    lim = [r for r in ok if r["steps"] >= 500]
    return dict(RStop=len(stop), RStopOk=sum(r["success"] >= 1 for r in stop), RStopPrec=f"{100 * sum(r['success'] >= 1 for r in stop) / len(stop):.1f}",
                RWrong=len(fs), RWrongNear=sum(r["final_dist"] <= 1.5 for r in fs),
                RWrongDist=f"{np.median([r['final_dist'] for r in fs]):.1f}", RLimit=sum(r["steps"] >= 500 for r in ok),
                RStepsSucc=f"{np.median([r['steps'] for r in stop if r['success'] >= 1]):.0f}",
                RCallsEp=f"{np.mean([r['vlm_calls'] for r in ok]):.0f}",
                RLimitAppr=sum("a" in r["modes"] for r in lim),
                RLimitRej=sum(1 for r in lim for e in r["events"] if isinstance(e, dict) and e.get("kind") == "reject"),
                RLimitExpl=f"{100 * np.mean([r['modes'].count('e') / len(r['modes']) for r in lim]):.0f}",
                RTimeStep=f"{sum(r['elapsed_s'] for r in ok) / sum(r['steps'] for r in ok):.2f}")


subsets(); probe(); outcomes(); forest(); episode()
facts = robothor()
with open(os.path.join(GEN, "thesis_numbers_fig.tex"), "w") as f:
    for k, v in facts.items():
        f.write(f"\\newcommand{{\\t{k}}}{{{v}}}\n")
print(facts)
