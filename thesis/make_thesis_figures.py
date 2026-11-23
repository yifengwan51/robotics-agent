"""Figures used only by the thesis, drawn from the generated numbers and the saved test records.

  ~/miniforge3/envs/embnav/bin/python thesis/make_thesis_figures.py   -> thesis/figures/th_*.pdf
Palette: agents keep the paper's colours (S2M orange, RGB-only light orange, EB prompt grey, oracle as a black line)
plus slate for the modular agent; orange never means anything but S2M. Episode outcomes use their own
categorical set (validated for colour-vision deficiency with the dataviz validator), and significance is shown by filled
vs open markers, so no colour carries two meanings across the thesis figures.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = os.path.join(ROOT, "thesis")
FIG, GEN = os.path.join(TH, "figures"), os.path.join(TH, "chapters", "generated")
TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_paper_figures import check  # noqa: E402

S2M, RGBO, EB, MOD, ORC = "#C55A11", "#F4B183", "#8C8C8C", "#44546A", "#0b0b0b"
# outcomes: success aqua; failures blue, magenta, violet, yellow, red, never orange (orange means S2M); adjacent CVD
# dE >= 9.6, normal-vision dE >= 20.8 (dataviz validator, light mode); approach failure stays neutral grey
OUT = {"success": "#1baf7a", "never seen": "#2a78d6", "lost after seen": "#e87ba4", "near miss": "#6250d6",
       "blocked": "#eda100", "oscillation": "#e34948", "approach failure": "#BFBFBF"}
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
              ("S2M", "SThree", S2M, ""), ("S2M, RGB only", "RGBThree", RGBO, "////")]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    x, w = np.arange(len(SUBS)), 0.19
    for i, (lab, key, c, h) in enumerate(series):
        v = [num(key + s) for s in SUBS]
        ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=lab, hatch=h, edgecolor="white" if not h else S2M, lw=0.4)
    for i, s in enumerate(SUBS):  # the oracle as a line over each group, as in the paper's Figure 2
        ax.plot([x[i] - 2.1 * w, x[i] + 2.1 * w], [num("Oracle" + s)] * 2, color=ORC, lw=1.0, solid_capstyle="butt",
                label="Oracle target" if i == 0 else None, zorder=3,
                path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    for i, (lab, key, c, h) in enumerate(series):  # values on the bars, as in the other result figures
        for xi, s_ in zip(x, SUBS):
            ax.text(xi + (i - 1.5) * w, 2.0, f"{num(key + s_):.0f}", ha="center", va="bottom",
                    fontsize=6.6, color="white" if c not in (RGBO,) else INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s_}\n(60 episodes)" for s_ in SUBS], color=INK)
    ax.set_ylabel("Success (%)", color=INK)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.17)
    save(fig, "th_subsets")


# 2. single-view probe: recall on views with the target against boxes on views without it (model identity: grey pair,
#    since orange means S2M elsewhere)
def probe():
    t = open(os.path.join(GEN, "supp_tab_probe.tex")).read()
    rows = {m.group(1): [float(v) for v in m.group(2).split("&")] for m in
            re.finditer(r"^(Qwen[\w.-]+) & ([\d. &]+)\\\\", t, re.M)}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    for ax, (title, sl) in zip(axes, (("Box on the visible target (higher is better)", slice(0, 4)),
                                     ("Box on a view without the target (lower is better)", slice(4, 8)))):
        x = np.arange(4)
        for i, (m, fc) in enumerate((("Qwen3-VL-4B", "#262626"), ("Qwen2.5-VL-3B", "white"))):
            v = rows[m][sl]
            ax.bar(x + (i - 0.5) * 0.36, v, 0.31, color=fc, label=m, edgecolor="#262626", lw=0.8)
            for xi, vi in zip(x + (i - 0.5) * 0.36, v):
                ax.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(SUBS[:4])
        ax.set_title(title, color=INK)
        ax.set_ylim(0, 95)
        style(ax)
    axes[1].tick_params(labelleft=False, length=0, axis="y")
    axes[0].set_ylabel("Views (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.8, bottom=0.12, wspace=0.06)
    save(fig, "th_probe")


# 3. outcomes of every agent: stacked bars over the 300 test episodes
def outcomes():
    t = open(os.path.join(GEN, "supp_tab_taxonomy.tex")).read()
    rows = [(m.group(1), [float(v) for v in m.group(2).split("&")]) for m in
            re.finditer(r"^([a-z ]+) & ([\d. &]+)\\\\", t, re.M)]
    agents = ["EB prompt\n(Qwen3)", "S2M\n(Qwen3)", "S2M, RGB only\n(Qwen3)", "EB prompt\n(Qwen2.5)", "S2M\n(Qwen2.5)"]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    left = np.zeros(len(agents))
    y = np.array([4.4, 3.4, 2.4, 1.0, 0.0])  # a gap between the two models
    for name, v in rows:
        v = np.array(v)
        ax.barh(y, v, 0.62, left=left, color=OUT[name], label=name, edgecolor="white", lw=0.6)
        for yi, li, vi in zip(y, left, v):
            if vi >= 12:
                ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if name in ("never seen", "near miss", "oscillation") else INK)
        left += v
    tot = {name: np.array(v) for name, v in rows}  # thin segments: all failure counts of each agent, right of its bar
    for i, yi in enumerate(y):
        small = [f"{n} {tot[n][i]:.1f}".replace(".0", "") for n in OUT if n != "success" and tot[n][i] < 12]
        if small:
            ax.text(304, yi, "\n".join(", ".join(small[j:j + 2]) for j in range(0, len(small), 2)), ha="left",
                    va="center", fontsize=6.2, color=INK2, linespacing=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(agents, fontsize=8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Episodes (of 300)")
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()  # legend in stack order, read left to right
    ax.legend(h, l, ncol=len(l), loc="lower center", bbox_to_anchor=(0.43, 1.0), frameon=False, handlelength=0.9,
              columnspacing=0.7, handletextpad=0.3, fontsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.7, top=0.88, bottom=0.17)
    save(fig, "th_outcomes")


# 4. ablation forest plot: paired difference from the full agent with 95% CI
def forest():
    groups = [("Components removed", [("w/o fusion", "Fusion"), ("w/o look-around", "Scan"), ("w/o exhaustion", "Exhaust"),
                                      ("w/o crop check", "Zoom")]),
              ("Sensing changed", [("odometry noise", "Odom"), ("RGB only, global scale", "Rgbglobal"),
                                   ("RGB only, floor scale", "Rgbfloor")])]
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    yi, ticks, labels = 0.0, [], []
    for g, items in groups:
        ax.text(-37.5, yi, g, fontsize=7.5, color=INK2, style="italic", va="center", ha="left")
        yi -= 1.0
        for lab, k in items:
            m, lo, hi = num(f"Abl{k}"), num(f"Abl{k}Lo"), num(f"Abl{k}Hi")
            sig = p_value(N[f"Abl{k}P"]) < 0.05
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot([m], [yi], "o", ms=5.5, mfc=INK if sig else "white", mec=INK, mew=1.1, zorder=3,
                    label=("Holm-corrected $p<0.05$" if sig else "not significant") if lab in ("w/o fusion", "w/o look-around") else None)
            ax.text(hi + 0.8, yi, f"{m:+.1f}".replace("-", "\u2212"), va="center", fontsize=7.5, color=INK)
            ticks.append(yi)
            labels.append(lab)
            yi -= 1.0
        yi -= 0.3
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(yi + 0.6, 0.6)
    ax.set_xlabel("Paired difference from the full agent (pp), 95% CI")
    ax.set_xlim(-38, 8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "\u2212")))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=True, facecolor="white", edgecolor="none", framealpha=1, handletextpad=0.3, borderaxespad=0.2)
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.26, right=0.99, top=0.97, bottom=0.16)
    save(fig, "th_forest")


# 5. one episode: camera frames with the VLM's box, and the evidence of every hypothesis over the steps
def episode():
    run = "s2m_Qwen3-VL-4B-Instruct-AWQ_simp4_r1"
    ep = json.load(open(os.path.join(TEST, run, "long_horizon", "episode_42.json")))
    rep = os.path.join(ROOT, "paper", "figures", "scene_replay", "s2m_qwen3_r1_long_42")
    man = json.load(open(os.path.join(rep, "manifest.json")))
    frames = {f["step"]: f["rgb"] for f in man["frames"]}
    t = ep["target"]
    near = lambda x, z: np.hypot(x - t["x"], z - t["z"]) < 0.5
    log = {e["step"]: e for e in ep["perception_log"]}
    show = [0, 2, 7, 13]
    fig = plt.figure(figsize=(6.3, 3.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.05)
    for i, s in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        img = np.asarray(Image.open(os.path.join(rep, frames[s])).convert("RGB"))
        e = log[s]
        cx = cy = 0
        if e.get("box") is not None and e["box"][2] - e["box"][0] < 40:  # small box: 2.5x crop around it
            b, side = e["box"], 200
            cx = int(min(max((b[0] + b[2]) / 2 - side / 2, 0), img.shape[1] - side))
            cy = int(min(max((b[1] + b[3]) / 2 - side / 2, 0), img.shape[0] - side))
            img = img[cy:cy + side, cx:cx + side]
        ax.imshow(img, interpolation="lanczos")
        if e.get("box") is not None:
            x0, y0, x1, y1 = e["box"][0] - cx, e["box"][1] - cy, e["box"][2] - cx, e["box"][3] - cy
            on = e["dets"] and near(*e["dets"][0][:2])
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.0,
                                       ec=S2M if on else "white",
                                       path_effects=[pe.Stroke(linewidth=3.6, foreground="white" if on else INK), pe.Normal()]))
        ax.set_title(f"step {s}" + (" (enlarged)" if cx or cy else ""), fontsize=8, color=INK)
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
        ax.plot([a for a, _ in tr], [b for _, b in tr], "--s", color=EB, ms=3, lw=1.2,
                label="look-alikes (two)" if j == 0 else None)
    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 2)))
    ax.text(13.75, 0.8, r"commit $\tau_\mathrm{c}$", fontsize=7.5, color=INK2, ha="left", va="center")
    # events of the episode, read from its perception log and plans
    first = next(e["step"] for e in ep["perception_log"] if any(near(h[0], h[1]) for h in e["hyps"]))
    commit = next(e["step"] for e in ep["perception_log"] if any(near(h[0], h[1]) and h[2] >= 0.8 for h in e["hyps"]))
    la = max((w, st) for tr in look.values() for st, w in tr)
    kw = dict(fontsize=7.2, color=INK, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.annotate("first box on the target", (first, wt[steps.index(first)]), (first - 1.9, 5.0), **kw)
    ax.annotate("committed", (commit, wt[steps.index(commit)]), (commit + 0.7, 0.2), **kw)
    ax.annotate("a look-alike passes $\\tau_\\mathrm{c}$;\nthe committed target is kept", (la[1], la[0]),
                (la[1] - 3.3, 2.6), **kw)
    for s in show:  # the steps of the frames above
        ax.axvline(s, color=GRID, lw=3, zorder=0)
    ax.set_xlim(-0.3, 13.6)
    ax.set_ylim(-0.4, 8.6)
    ax.set_xticks(range(0, 14))
    for st, lab in zip(range(0, 14), ax.get_xticklabels()):
        if st in show:
            lab.set_fontweight("bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Evidence $w$")
    style(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.11)
    save(fig, "th_episode")


# 6. RoboTHOR: how episodes end
def robothor():
    R = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "robothor", "val",
                                                          "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json"))]
    ok = [r for r in R if not r.get("error")]
    stop = [r for r in ok if r["modes"].endswith("S")]
    parts = [("success: stops at the target", sum(r["success"] >= 1 for r in stop), OUT["success"], ""),
             ("failure: stops away from the target", sum(r["success"] < 1 for r in stop), "#595959", ""),
             ("failure: step limit (500)", sum(r["steps"] >= 500 for r in ok), "white", "////"),
             ("start pose rejected by the simulator", len(R) - len(ok), "white", "")]
    fig, ax = plt.subplots(figsize=(6.3, 1.45))
    left = 0
    for lab, v, c, h in parts:
        ax.barh([0], [v], 0.55, left=left, color=c, hatch=h, edgecolor="#595959" if c == "white" else "white", lw=0.7,
                label=lab)
        if v > 60:
            ax.text(left + v / 2, 0, f"{v} ({100 * v / len(R):.1f}%)", ha="center", va="center", fontsize=8,
                    color="white" if c == "#595959" else INK,
                    bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none") if h else None)
        else:  # a sliver: its count beside the bar end
            ax.annotate(f"{v} ({100 * v / len(R):.1f}%)", (left + v / 2, 0.3), (left + v / 2 - 60, 0.52), ha="center",
                        va="bottom", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        left += v
    ax.set_xlim(0, len(R))
    ax.set_xticks([0, 300, 600, 900, 1200, 1500, len(R)])
    ax.set_ylim(-0.4, 0.75)
    ax.set_yticks([])
    ax.set_xlabel(f"RoboTHOR validation episodes (of {len(R)})")
    style(ax, ygrid=False)
    ax.spines["left"].set_visible(False)
    h, l = ax.get_legend_handles_labels()  # row-wise in bar order
    fig.legend([h[0], h[2], h[1], h[3]], [l[0], l[2], l[1], l[3]], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               frameon=False, handlelength=1.4)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.66, bottom=0.33)
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


# 7. RoboTHOR: success within a step budget, and the presence answer behind every stop and rejection
def robothor_budget():
    R = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "robothor", "val",
                                                          "s2m_Qwen3-VL-4B-Instruct-AWQ_final", "*.json"))]
    ok = [r for r in R if not r.get("error")]
    succ = sorted(r["steps"] for r in ok if r["success"] >= 1)
    t = np.arange(0, 501)
    curve = [100 * np.searchsorted(succ, b, side="right") / len(R) for b in t]
    ev = [e for r in ok for e in r.get("events", []) if isinstance(e, dict)]
    stop_ok = [e["presence"] for r in ok for e in r.get("events", []) if isinstance(e, dict) and e.get("kind") == "stop"
               and r["success"] >= 1]
    stop_bad = [e["presence"] for r in ok for e in r.get("events", []) if isinstance(e, dict) and e.get("kind") == "stop"
                and r["success"] < 1]
    rej = [e["presence"] for e in ev if e.get("kind") == "reject"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    ax = axes[0]
    ax.plot(t, curve, color=S2M, lw=1.6)
    for b in (100, 200):
        ax.plot([b], [curve[b]], "o", color=S2M, ms=3.5)
        ax.text(b + 8, curve[b] - 1.5, f"{curve[b]:.1f}% by step {b}", fontsize=7.5, color=INK, va="top")
    ax.axhline(curve[-1], color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(495, curve[-1] + 1.2, f"{curve[-1]:.1f}% at the 500-step limit", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Step budget")
    ax.set_ylabel("Success (%)")
    ax.set_title("(a) Success within a step budget", loc="left", color=INK)
    style(ax)
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    ax.hist([rej, stop_bad, stop_ok], bins=bins, stacked=True, color=["#BFBFBF", "#595959", OUT["success"]],
            label=[f"rejected ({len(rej)})", f"stopped, failed ({len(stop_bad)})", f"stopped, success ({len(stop_ok)})"],
            edgecolor="white", lw=0.5)
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.text(0.52, ax.get_ylim()[1] * 0.97, "stop threshold", fontsize=7.5, color=INK, va="top")
    ax.set_xlabel("Presence at the stop check")
    ax.set_ylabel("Checks")
    ax.set_title("(b) The stop check", loc="left", color=INK)
    style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=7.2, handlelength=1.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.18, wspace=0.32)
    save(fig, "th_robothor_budget")
    return dict(RBudHundred=f"{curve[100]:.1f}", RBudTwoHundred=f"{curve[200]:.1f}", RRejN=len(rej),
                RRejEp=sum(any(isinstance(e, dict) and e.get("kind") == "reject" for e in r.get("events", [])) for r in ok),
                RRejEpOk=sum(any(isinstance(e, dict) and e.get("kind") == "reject" for e in r.get("events", []))
                             and r["success"] >= 1 for r in ok),
                RStepsPninety=f"{np.percentile(succ, 90):.0f}",
                RCheckN=len(rej) + len(stop_ok) + len(stop_bad),
                RCheckSure=f"{100 * np.mean([(v <= 0.1 or v >= 0.9) for v in rej + stop_ok + stop_bad]):.0f}")


subsets(); probe(); outcomes(); forest(); episode()
facts = robothor()
facts.update(robothor_budget())
with open(os.path.join(GEN, "thesis_numbers_fig.tex"), "w") as f:
    for k, v in facts.items():
        f.write(f"\\newcommand{{\\t{k}}}{{{v}}}\n")
print(facts)
