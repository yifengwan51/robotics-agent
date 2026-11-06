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
S2M, RGBO, EB, INK, INK2, MUTED, GRID = "#C55A11", "#F4B183", "#7F7F7F", "#0b0b0b", "#595959", "#BFBFBF", "#E7E6E6"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7.2, "axes.titlesize": 7.5, "axes.labelsize": 7.2, "xtick.labelsize": 7.0,
                     "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "pdf.fonttype": 42})


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
    pts = [ln.get_transform().transform(np.column_stack(ln.get_data())) for ln in ax.lines]
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
    b = DATA["bars"]
    subsets = DATA["subsets"] + ["Avg."]
    panels = [("Qwen3-VL-4B", [(b["eb3"], EB, "EB prompt"), (b["s2m3"], S2M, "S2M (ours)"),
                               (b["rgb3"], RGBO, "S2M, RGB only")]),
              ("Qwen2.5-VL-3B", [(b["eb25"], EB, "EB prompt"), (b["s2m25"], S2M, "S2M (ours)")])]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.45), sharex=True)
    w, x = 0.3, np.arange(len(subsets))
    for ax, (title, series) in zip(axes, panels):
        off = [(k - (len(series) - 1) / 2) * w for k in range(len(series))]
        for (v, c, lab), o in zip(series, off):
            ax.bar(x + o, v, w, color=c, label=lab, zorder=2, edgecolor=S2M if c == RGBO else "white",
                   linewidth=0.5 if c == RGBO else 0.6)
        half = len(series) / 2 * w
        for i, o in enumerate(b["oracle"]):
            ax.plot([i - half - .03, i + half + .03], [o, o], color=INK, lw=0.9, zorder=3,
                    label="oracle target" if i == 0 else None)
        for (v, c, _), o in zip(series, off):
            i = len(subsets) - 1
            inside = v[i] > 30
            ax.text(i + o, v[i] - 2 if inside else v[i] + 1.5, f"{v[i]:.1f}", rotation=90, ha="center",
                    va="top" if inside else "bottom", fontsize=6.5, color=INK,
                    zorder=4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        style(ax)
        ax.set_ylabel("success (%)", color=INK2)
        ax.set_title(title, loc="left", color=INK, pad=2)
        ax.axvline(len(subsets) - 1.5, color=GRID, lw=0.8, zorder=0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subsets)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[1:] + h[:1], lab[1:] + lab[:1], loc="upper center", ncol=2, frameon=False, handlelength=1.3,
               columnspacing=1.2, bbox_to_anchor=(0.55, 1.01))
    fig.subplots_adjust(left=0.13, right=0.995, top=0.8, bottom=0.1, hspace=0.42)
    check(fig, "results")
    fig.savefig(os.path.join(FIG, "results.pdf"))


def analysis():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.0]))
    # (a) success within a step budget
    ax = axes[0]
    t = np.arange(21)
    bud = DATA["budget"]
    for key, c, lab, ls in (("oracle", INK, "oracle target", (0, (1, 1.5))), ("s2m3", S2M, "S2M", "-"),
                            ("rgb3", S2M, "S2M, RGB only", (0, (4, 2))), ("eb3", EB, "EB prompt", "-")):
        ax.plot(t, bud[key], color=c, lw=1.4 if key != "oracle" else 1.0, ls=ls, label=lab, zorder=3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("step budget")
    ax.set_ylabel("episodes solved (%)")
    style(ax, grid="both")
    ax.set_title("(a) success within a step budget", loc="left", pad=3)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.2, wspace=0.42)
    fit_legend(fig, ax, ["lower right", ("lower right", {"ncol": 2, "columnspacing": 0.8}), "upper left",
                         ("upper left", {"ncol": 2, "columnspacing": 0.8})],
               headroom=(1.0, 1.15, 1.3, 1.45), frameon=False, handlelength=1.6, borderaxespad=0.2)
    # (b) failure modes, EB prompt vs S2M (episodes of 300)
    ax = axes[1]
    cats = [("never_seen", "never seen"), ("lost_after_seen", "lost after seen"), ("near_miss", "near miss"),
            ("approach_fail", "approach failure"), ("blocked", "blocked"), ("oscillation", "oscillation")]
    tx = DATA["taxonomy"]
    y = np.arange(len(cats))[::-1]
    h = 0.36
    for k, (key, c, lab) in enumerate((("eb3", EB, "EB prompt"), ("s2m3", S2M, "S2M"))):
        v = [tx[key][cname] for cname, _ in cats]
        ax.barh(y + (h / 2 if k == 0 else -h / 2), v, h, color=c, label=lab, zorder=2, edgecolor="white",
                linewidth=0.5)
        for yy, vv in zip(y + (h / 2 if k == 0 else -h / 2), v):
            ax.text(vv + 0.8, yy, f"{vv:.0f}" if float(vv).is_integer() else f"{vv:.1f}", va="center", ha="left",
                    fontsize=6.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in cats])
    ax.set_xlabel("failed episodes (of 300)")
    ax.set_xlim(0, max(max(tx["eb3"][c] for c, _ in cats), max(tx["s2m3"][c] for c, _ in cats)) * 1.18)
    style(ax, grid="x")
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    ax.set_title("(b) why episodes fail", loc="left", pad=3)
    # (c) lifted range / true range of target detections
    ax = axes[2]
    bins = np.linspace(0.4, 1.6, 49)
    for key, c, lab, ls in (("depth", S2M, "depth sensor", "-"), ("floor", S2M, "floor scale", (0, (4, 2))),
                            ("global", EB, "global scale", (0, (1, 1.5)))):
        v = np.asarray(DATA["range_ratio"][key])
        ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1.3, ls=ls, label=lab, zorder=3)
        ax.axvline(np.median(v), color=c, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    ax.axvline(1.0, color=INK, lw=0.6, zorder=1)
    ax.set_xlim(0.4, 1.6)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_xlabel("lifted range / true range")
    ax.set_ylabel("density")
    style(ax, grid="y")
    ax.set_title("(c) range of lifted target boxes", loc="left", pad=3)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.2, wspace=0.42)
    fit_legend(fig, ax, ["upper right"], headroom=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2), ncol=1,
               frameon=False, handlelength=1.3, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.15)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.2, wspace=0.42)
    check(fig, "analysis")
    fig.savefig(os.path.join(FIG, "analysis.pdf"))


# backbone size (B parameters) of the open models in tab:main; closed models go to the API column
SIZE = {"InternVL3-78B": 78, "Qwen2.5-VL-72B": 72, "InternVL3-8B": 8, "Qwen2.5-VL-7B": 7, "Qwen3-VL-8B": 8,
        "Qwen3-VL-32B": 32, "Qwen3.6-35B-A3B": 35, "Qwen3.8-27B": 27, "Qwen3.5-4B": 4, "Qwen3-VL-4B": 4}
API = 170


def landscape():
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    marks = {"Prompted MLLMs (EmbodiedBench)": ("o", MUTED, MUTED, "prompted MLLM (reported)"),
             "Training-free agents": ("s", "white", INK2, "training-free agent (reported)"),
             "Fine-tuned": ("^", "white", INK2, "fine-tuned agent (reported)")}
    shift = {"Prompted MLLMs (EmbodiedBench)": 0.93, "Training-free agents": 1.07, "Fine-tuned": 0.93}
    seen = set()
    for group, name, model, avg in DATA["published"]:
        m, fc, ec, lab = marks[group]
        x = SIZE.get(model, API) * shift[group]
        ax.scatter(x, avg, s=16, marker=m, facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3,
                   label=lab if group not in seen else None)
        seen.add(group)
    b = DATA["bars"]
    for eb, s2m, size in (("eb3", "s2m3", 4), ("eb25", "s2m25", 3)):
        x = size * 1.07
        ax.annotate("", xy=(x, b[s2m][-1] - 2.2), xytext=(x, b[eb][-1] + 2.2),
                    arrowprops=dict(arrowstyle="-|>", color=S2M, lw=0.9, mutation_scale=7), zorder=2)
        ax.scatter(x, b[eb][-1], s=16, marker="o", facecolor="white", edgecolor=EB, linewidth=0.9, zorder=3,
                   label="EB prompt (our run)" if size == 4 else None)
        ax.scatter(x, b[s2m][-1], s=46, marker="*", color=S2M, edgecolor="white", linewidth=0.3, zorder=4,
                   label="S2M (ours)" if size == 4 else None)
        ax.text(x * 1.12, (b[eb][-1] + b[s2m][-1]) / 2, f"+{b[s2m][-1] - b[eb][-1]:.1f}", color=S2M,
                fontsize=6.6, va="center", ha="left")
    for name, model, dx, dy, ha in (("Tool library", "Qwen3-VL-8B", 1.12, 0, "left"),
                                    ("EMHO", "Qwen3.8-27B", 1.18, 0, "left")):
        avg = next(a for g, n, mo, a in DATA["published"] if n.startswith(name) and mo == model)
        ax.text(SIZE[model] * (0.93 if ha == "right" else 1.07) * dx, avg + dy, name, fontsize=6.4, color=INK2,
                va="center", ha=ha)
    ax.axvline(115, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(2.4, 230)
    ax.set_xticks([3, 10, 30, 80, API])
    ax.set_xticklabels(["3B", "10B", "30B", "80B", "API"])
    ax.minorticks_off()
    ax.set_ylim(10, 95)
    ax.set_xlabel("size of the model (parameters)")
    ax.set_ylabel("EB-Navigation success (%)")
    style(ax, grid="y")
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in ("S2M (ours)", "EB prompt (our run)", "prompted MLLM (reported)",
                                    "training-free agent (reported)", "fine-tuned agent (reported)")]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=2, frameon=False,
               handlelength=1.0, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.54, 1.01))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.77, bottom=0.15)
    check(fig, "landscape")
    fig.savefig(os.path.join(FIG, "landscape.pdf"))


def robothor_cats():
    cats = sorted(DATA["robothor_cats"].items(), key=lambda kv: kv[1][0])
    names = [{"BasketBall": "basketball", "BaseballBat": "baseball bat"}.get(k, "".join(
        " " + c.lower() if c.isupper() and i else c.lower() for i, c in enumerate(k))) for k, _ in cats]
    sr = [v[0] for _, v in cats]
    spl = [v[1] for _, v in cats]
    fig, ax = plt.subplots(figsize=(3.35, 1.55))
    y = np.arange(len(cats))
    ax.barh(y, sr, 0.62, color=S2M, zorder=2, label="SR")
    ax.scatter(spl, y, marker="|", s=40, color=INK, linewidth=1.3, zorder=3, label="SPL")
    for yy, v in zip(y, sr):
        ax.text(v + 1.2, yy, f"{v:.0f}", va="center", ha="left", fontsize=6.4, color=INK)
    ax.axvline(DATA["robothor_sr"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(DATA["robothor_sr"] + 1, len(cats) - 0.35, f"all {DATA['robothor_sr']:.1f}", fontsize=6.4,
            color=INK2, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, len(cats) + 0.5)
    ax.set_xlabel("RoboTHOR validation (%)")
    style(ax, grid="x")
    ax.legend(loc="lower right", frameon=False, handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(left=0.22, right=0.965, top=0.99, bottom=0.215)
    check(fig, "robothor_cats")
    fig.savefig(os.path.join(FIG, "robothor_cats.pdf"))


if __name__ == "__main__":
    if DATA.get("draft"):
        print("NOTE: figure data is a DRAFT (FP8 Qwen3 baseline missing)")
    teaser()
    analysis()  # the architecture figure: scripts/make_arch_figure.py
    landscape()
    robothor_cats()
