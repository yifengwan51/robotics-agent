"""Figures of the paper from paper/figures/paper_data.json (scripts/paper_assets.py) and one replayed episode.

  python scripts/make_paper_figures.py      (env embnav: matplotlib)
    -> paper/latex/fig/results.pdf      teaser: success per subset, EB prompt vs S2M, both models
    -> paper/latex/fig/analysis.pdf     (a) success vs step budget (b) failure modes (c) range bias of lifted targets

Every figure is checked for overlapping or clipped text before it is written.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "paper", "latex", "fig")
DATA = json.load(open(os.path.join(ROOT, "paper", "figures", "paper_data.json")))
REPLAY = os.path.join(ROOT, "paper", "figures", "scene_replay", "s2m_qwen3_r1_long_42")
EPISODE = os.path.join(ROOT, "runs", "eb_nav", "test", "s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1", "long_horizon",
                       "episode_42.json")
# categorical slots 1-3 of the validated palette (light mode); the oracle and unused variants stay neutral
# one colour per agent in every figure of the paper (Office ladders, as in the architecture figure):
# S2M orange.deep, S2M RGB-only orange.soft (dashed / outlined), EB prompt grey.dark-mid, oracle black
S2M, RGBO, EB, INK, INK2, MUTED, GRID = "#C55A11", "#F4B183", "#7F7F7F", "#0b0b0b", "#404040", "#BFBFBF", "#E7E6E6"
# THESIS=1: the full-width figures are redrawn at the thesis text width (6.3 in) with text scaled by K, so that
# they read at about 9 pt next to 12 pt body text; output goes to *_thesis.pdf and the paper's files are untouched
THESIS = os.environ.get("THESIS") == "1"
K = 1.3 if THESIS else 1.0
SUF = "_thesis" if THESIS else ""
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7.2 * K, "axes.titlesize": 7.5 * K, "axes.labelsize": 7.2 * K,
                     "xtick.labelsize": 7.0 * K, "ytick.labelsize": 7.0 * K, "legend.fontsize": 7.0 * K,
                     "pdf.fonttype": 42})


def style(ax, grid="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, length=2, width=0.6)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def legend_covers(lg, ax, r):
    """True if the legend box contains any point of a line, histogram, bar, or marker of `ax`."""
    box = lg.get_window_extent(r)
    # data lines only: reference lines (axhline/axvline, no legend label) may pass behind a legend
    pts = [ln.get_transform().transform(np.column_stack(ln.get_data())) for ln in ax.lines
           if not ln.get_label().startswith("_")]
    pts += [p.get_path().transformed(p.get_transform()).vertices for p in ax.patches]
    pts += [c.get_offset_transform().transform(c.get_offsets()) for c in ax.collections if len(c.get_offsets())]
    for xy in pts:
        xy = np.asarray(xy, float)
        if len(xy) > 1:  # densify segments so a thin peak crossing the box is caught
            t = np.linspace(0, 1, 12)[:, None, None]
            xy = (xy[:-1] * (1 - t) + xy[1:] * t).reshape(-1, 2)
        if ((xy[:, 0] > box.x0 + 1) & (xy[:, 0] < box.x1 - 1) & (xy[:, 1] > box.y0 + 1) & (xy[:, 1] < box.y1 - 1)).any():
            return True
    return False


def fit_legend(fig, ax, locs, headroom=(1.0,), **kw):
    """Place the legend at the first location (with the least extra y headroom) where it covers no data."""
    y0, y1 = ax.get_ylim()
    for h in headroom:
        ax.set_ylim(y0, y0 + (y1 - y0) * h)
        for loc in locs:
            loc, extra = loc if isinstance(loc, tuple) else (loc, {})
            lg = ax.legend(loc=loc, **{**kw, **extra})
            fig.canvas.draw()
            if not legend_covers(lg, ax, fig.canvas.get_renderer()):
                return
    raise SystemExit(f"no legend position free of data in '{ax.get_title(loc='left')}'")


def check(fig, name):
    """Fail on overlapping text or text outside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        items += [("text " + t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        ticks = [t for t in ax.get_xticklabels() if x0 - 1e-9 <= t.get_position()[0] <= x1 + 1e-9] + \
                [t for t in ax.get_yticklabels() if y0 - 1e-9 <= t.get_position()[1] <= y1 + 1e-9]
        items += [("tick " + t.get_text(), t.get_window_extent(r)) for t in ticks if t.get_text().strip()]
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title):
            if lab.get_text().strip():
                items.append(("label " + lab.get_text(), lab.get_window_extent(r)))
    for lg in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        items.append(("legend", lg.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    bad = []
    for lg in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        for ax in fig.axes:
            if legend_covers(lg, ax, r):
                bad.append(f"legend covers data in '{ax.get_title(loc='left')}'")
    for i, (a, ba) in enumerate(items):
        if ba.x0 < -1 or ba.y0 < -1 or ba.x1 > W + 1 or ba.y1 > H + 1:
            bad.append(f"clipped: {a}")
        for b, bb in items[i + 1:]:
            if a.startswith("tick") and b.startswith("tick"):
                continue
            if ba.overlaps(bb) and ba.width > 0 and bb.width > 0:
                ox, oy = min(ba.x1, bb.x1) - max(ba.x0, bb.x0), min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                if ox > 0.3 and oy > 0.3:  # any real overlap (pixels at 100 dpi)
                    bad.append(f"overlap: {a} | {b}")
    if bad:
        raise SystemExit(f"{name}: " + "; ".join(bad))
    print(f"{name}: layout ok ({len(items)} text items)")


def teaser():
    """Success per subset with the same model: EB prompt vs S2M (and RGB-only S2M for Qwen3), oracle as a tick."""
    b = DATA["bars"]
    subsets = DATA["subsets"] + ["Avg."]
    x = np.arange(len(subsets), dtype=float)
    x[-1] += 0.35  # the average stands apart from the five subsets
    w = 0.27
    panels = [("Qwen3-VL-4B", [("eb3", EB, "EB prompt"), ("s2m3", S2M, "S2M (ours)"), ("rgb3", RGBO, "S2M, RGB only")]),
              ("Qwen2.5-VL-3B", [("eb25", EB, "EB prompt"), ("s2m25", S2M, "S2M (ours)")])]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.45), sharex=True)
    for ax, (title, series) in zip(axes, panels):
        off = [(k - (len(series) - 1) / 2) * w for k in range(len(series))]
        for (key, c, lab), o in zip(series, off):
            ax.bar(x + o, b[key], w * 0.94, color=c, label=lab, zorder=2,
                   edgecolor=S2M if c == RGBO else "none", linewidth=0.6 if c == RGBO else 0)
        for i, o in enumerate(b["oracle"]):  # same width in both panels
            ax.plot([x[i] - 1.5 * w, x[i] + 1.5 * w], [o, o], color=INK, lw=1.0, solid_capstyle="butt", zorder=3,
                    label="oracle target" if i == 0 else None,
                    path_effects=[pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])  # stays visible on a bar top
        i = len(subsets) - 1
        for (key, c, _), o in zip(series, off):  # values of the average, inside tall bars and above short ones
            v = b[key][i]
            inside = v > 32
            ax.text(x[i] + o, v - 4 if inside else v + 3, f"{v:.1f}", rotation=90, ha="center",
                    va="top" if inside else "bottom", fontsize=6.6 * K, zorder=4,
                    color="white" if inside and c != RGBO else INK)
        ax.axvspan(x[-1] - 0.55, x[-1] + 0.55, color="#F5F5F4", zorder=0, lw=0)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        style(ax)
        ax.set_title(title, loc="left", color=INK, pad=3, fontsize=7.4 * K)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subsets)
    axes[-1].set_xlim(x[0] - 0.6, x[-1] + 0.6)
    fig.text(0.012, (0.86 + 0.09) / 2, "success (%)", rotation=90, va="center", ha="left", color=INK2, fontsize=7.2 * K)
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(k) for k in ("S2M (ours)", "S2M, RGB only", "EB prompt", "oracle target")]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="upper center", ncol=4, frameon=False,
               handlelength=1.1, columnspacing=0.8, handletextpad=0.35, bbox_to_anchor=(0.55, 1.01))
    fig.subplots_adjust(left=0.11, right=0.995, top=0.86, bottom=0.09, hspace=0.38)
    check(fig, "results")
    fig.savefig(os.path.join(FIG, "results.pdf"))


def analysis():
    if THESIS:  # thesis page: (a) and (b) side by side, (c) below at the same width as (a)
        fig = plt.figure(figsize=(6.3, 4.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.62, wspace=0.62)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1.0, 1.1, 1.0]))
    # (a) success within a step budget
    ax = axes[0]
    t = np.arange(21)
    bud = DATA["budget"]
    for key, c, lab, ls, lw in (("oracle", INK, "oracle target", (0, (3, 1.6)), 1.0), ("s2m3", S2M, "S2M", "-", 1.5),
                                ("rgb3", S2M, "S2M, RGB only", (0, (1.2, 1.2)), 1.5), ("eb3", EB, "EB prompt", "-", 1.5)):
        ax.plot(t, bud[key], color=c, lw=lw, ls=ls, label=lab, zorder=3)
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("step budget")
    ax.set_ylabel("episodes solved (%)")
    style(ax, grid="both")
    ax.set_title("(a) success within a step budget", loc="left", pad=3)
    if THESIS:
        fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.1)
    else:
        fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.21, wspace=0.62)
    # no episode is solved before step 7, so the upper left of the panel is free for the legend
    fit_legend(fig, ax, ["upper left", ("upper left", {"borderaxespad": 0.0}), ("upper left", {"ncol": 2, "columnspacing": 0.6}),
                         ("lower right", {"bbox_to_anchor": (1.0, 0.02)})],
               headroom=(1.0, 1.12, 1.25) if THESIS else (1.0,),
               frameon=False, handlelength=1.4, labelspacing=0.2, borderaxespad=0.2, handletextpad=0.4, fontsize=6.4 * K)
    # (b) failure modes, EB prompt vs S2M (episodes of 300)
    ax = axes[1]
    cats = [("never_seen", "never seen"), ("lost_after_seen", "lost after seen"), ("near_miss", "near miss"),
            ("approach_fail", "approach failure"), ("blocked", "blocked"), ("oscillation", "oscillation")]
    tx = DATA["taxonomy"]
    y = np.arange(len(cats))[::-1]
    h = 0.38
    for k, (key, c, lab) in enumerate((("eb3", EB, "EB prompt"), ("s2m3", S2M, "S2M"))):
        v = [tx[key][cname] for cname, _ in cats]
        yy = y + (h / 2 if k == 0 else -h / 2)
        ax.barh(yy, v, h * 0.92, color=c, label=lab, zorder=2, lw=0)
    for y0, (cname, _) in zip(y, cats):  # one label per row, EB prompt / S2M (run means rounded)
        e, m = tx["eb3"][cname], tx["s2m3"][cname]
        ax.text(max(e, m) + 0.9, y0, f"{e:.0f} / {m:.0f}", va="center", ha="left", fontsize=6.4 * K, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in cats])
    ax.set_xlabel("failed episodes (of 300)")
    ax.set_xlim(0, max(max(tx["eb3"][c] for c, _ in cats), max(tx["s2m3"][c] for c, _ in cats)) * 1.25)
    style(ax, grid="x")
    fit_legend(fig, ax, ["lower right", "center right"], frameon=False, handlelength=1.0, handletextpad=0.35,
               borderaxespad=0.1, fontsize=6.4 * K)
    ax.set_title("(b) why episodes fail", loc="left", pad=3)
    # (c) lifted range / true range of target detections, as cumulative distributions
    ax = axes[2]
    for key, c, lab, ls in (("depth", S2M, "depth sensor", "-"), ("floor", S2M, "floor scale", (0, (1.2, 1.2))),
                            ("global", EB, "global scale", "-")):
        v = np.sort(np.asarray(DATA["range_ratio"][key]))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.5, ls=ls, label=lab, zorder=3)
    ax.axvline(1.0, color=INK, lw=0.6, zorder=1)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(1.43, 0.53, "median", fontsize=6.4 * K, color=INK2, va="bottom", ha="right")
    ax.text(0.98, 0.97, "exact", fontsize=6.4 * K, color=INK, va="top", ha="right",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=4)
    ax.set_xlim(0.45, 1.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("lifted range / true range")
    ax.set_ylabel("fraction of detections")
    style(ax, grid="both")
    ax.set_title("(c) range of lifted target boxes", loc="left", pad=3)
    if THESIS:  # (c) has the space to its right for its legend
        ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), ncol=1, frameon=False, handlelength=1.6,
                  handletextpad=0.4, labelspacing=0.4, fontsize=6.4 * K)
    else:
        fit_legend(fig, ax, ["upper left", "lower right", ("lower right", {"borderaxespad": 0.0}),
                             ("center right", {"bbox_to_anchor": (1.0, 0.3)}), ("lower right", {"fontsize": 6.4 * K * 0.85})],
                   headroom=(1.0,), ncol=1, frameon=False,
                   handlelength=1.6, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
    check(fig, "analysis")
    fig.savefig(os.path.join(FIG, f"analysis{SUF}.pdf"))


# backbone size (B parameters) of the open models in tab:main; closed models go to the API column
SIZE = {"InternVL3-78B": 78, "Qwen2.5-VL-72B": 72, "InternVL3-8B": 8, "Qwen2.5-VL-7B": 7, "Qwen3-VL-8B": 8,
        "Qwen3-VL-32B": 32, "Qwen3.6-35B-A3B": 35, "Qwen3.8-27B": 27, "Qwen3.5-4B": 4, "Qwen3-VL-4B": 4}
API = 190


def landscape():
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    marks = {"Prompted MLLMs (EmbodiedBench)": ("o", "#8C8C8C", "#8C8C8C", "prompted MLLM"),
             "Training-free agents": ("s", "white", INK2, "training-free agent"),
             "Fine-tuned": ("^", "white", INK2, "fine-tuned agent")}
    shift = {"Prompted MLLMs (EmbodiedBench)": 0.88, "Training-free agents": 1.12, "Fine-tuned": 0.88}
    seen, api = set(), 0
    for group, name, model, avg in DATA["published"]:
        m, fc, ec, lab = marks[group]
        if model in SIZE:
            same = [n2 for g2, n2, m2, _ in DATA["published"] if g2 == group and m2 == model]
            xx = SIZE[model] * shift[group] * (1 + 0.16 * (same.index(name) - (len(same) - 1) / 2))
        else:  # closed models: spread the API column so equal scores do not hide each other
            xx = API * (0.74 + 0.085 * api)  # one slot per closed-model result
            api += 1
        ax.scatter(xx, avg, s=13, marker=m, facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3,
                   label=lab + " (reported)" if group not in seen else None)
        seen.add(group)
    b = DATA["bars"]
    for eb, s2m, size, tag, side in (("eb3", "s2m3", 4, "4B", "right"), ("eb25", "s2m25", 3, "3B", "left")):
        xx = size * (1.1 if side == "right" else 0.92)
        lo, hi = b[eb][-1], b[s2m][-1]
        ax.annotate("", xy=(xx, hi - 2.4), xytext=(xx, lo + 2.4),
                    arrowprops=dict(arrowstyle="-|>", color=S2M, lw=0.9, mutation_scale=7), zorder=2)
        ax.scatter(xx, lo, s=17, marker="o", facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4,
                   label="EB prompt (our run)" if size == 4 else None)
        ax.scatter(xx, hi, s=50, marker="*", color=S2M, edgecolor="white", linewidth=0.3, zorder=5,
                   label="S2M (ours)" if size == 4 else None)
        # both gains beside their arrows, the 3B one low enough to stay clear of the fine-tuned 4B agents
        ax.text(xx * 1.16, (lo + hi) / 2 if side == "right" else lo + 15, f"{tag}\n+{hi - lo:.1f} pp", color=S2M,
                fontsize=6.4 * K, va="center", ha="left", linespacing=1.0)
    for name, model in (("Tool library", "Qwen3-VL-8B"), ("EMHO", "Qwen3.8-27B")):
        avg = next(a for g, n, mo, a in DATA["published"] if n.lower().startswith(name.lower()) and mo == model)
        ax.text(SIZE[model] * 1.12 * 1.24, avg, name, fontsize=6.4 * K, color=INK2, va="center", ha="left")
    ax.axvline(120, color=MUTED, lw=0.7, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(2.0, 290)
    ax.set_xticks([3, 10, 30, 80, API])
    ax.set_xticklabels(["3B", "10B", "30B", "80B", "API"])
    ax.set_yticks([20, 40, 60, 80])
    ax.minorticks_off()
    ax.set_ylim(8, 90)
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("EB-Navigation success (%)")
    style(ax, grid="y")
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(k) for k in ("S2M (ours)", "EB prompt (our run)", "prompted MLLM (reported)",
                                    "training-free agent (reported)", "fine-tuned agent (reported)")]
    fig.subplots_adjust(left=0.135, right=0.99, top=0.98, bottom=0.17)
    lg = ax.legend([h[i] for i in order], [lab[i] for i in order], loc="lower right", ncol=1, frameon=False,
                   handlelength=1.0, handletextpad=0.4, labelspacing=0.22, borderaxespad=0.15, fontsize=6.6 * K,
                   bbox_to_anchor=(0.8, 0.0))
    check(fig, "landscape")
    fig.savefig(os.path.join(FIG, "landscape.pdf"))


def robothor_cats():
    cats = sorted(DATA["robothor_cats"].items(), key=lambda kv: kv[1][0])
    names = [{"BasketBall": "basketball", "BaseballBat": "baseball bat"}.get(k, "".join(
        " " + c.lower() if c.isupper() and i else c.lower() for i, c in enumerate(k))) for k, _ in cats]
    sr = [v[0] for _, v in cats]
    spl = [v[1] for _, v in cats]
    sr_all = DATA["robothor_sr"]
    spl_all = float(open(os.path.join(ROOT, "paper", "latex", "sec", "numbers.tex")).read()
                    .split("\\newcommand{\\nRoboSPL}{")[1].split("}")[0])
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    y = list(np.arange(len(cats))) + [len(cats) + 0.35]  # the pooled row sits apart, on top
    sr, spl, names = sr + [sr_all], spl + [spl_all], names + ["all"]
    ax.barh(y, sr, 0.78, color="#F4B183", zorder=2, lw=0)
    ax.barh(y, spl, 0.38, color=S2M, zorder=3, lw=0)
    for yy, a, p in zip(y, sr, spl):
        ax.text(a + 1.2, yy, f"{a:.0f} / {p:.0f}" if yy < len(cats) else f"{a:.1f} / {p:.1f}", va="center",
                ha="left", fontsize=6.3 * K, color=INK, zorder=5,
                fontweight="bold" if yy >= len(cats) else "normal")
    ax.axhline(len(cats) - 0.33, color=INK2, lw=0.4, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=6.4 * K)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, len(cats) + 0.85)
    ax.set_xlabel("RoboTHOR validation (%)")
    style(ax, grid="x")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#F4B183", lw=0), Patch(color=S2M, lw=0)], ["SR (first value)", "SPL (second value)"],
              loc="lower right", frameon=False, handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(left=0.2, right=0.965, top=0.99, bottom=0.215)
    check(fig, "robothor_cats")
    fig.savefig(os.path.join(FIG, "robothor_cats.pdf"))


if __name__ == "__main__":
    if DATA.get("draft"):
        print("NOTE: figure data is a DRAFT (FP8 Qwen3 baseline missing)")
    if THESIS:  # only the full-width figure needs a thesis variant; the column figures are reused as they are
        analysis()
        sys.exit(0)
    teaser()
    analysis()  # the architecture figure: scripts/make_arch_figure.py
    landscape()
    robothor_cats()
