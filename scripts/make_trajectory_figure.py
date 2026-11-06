"""Paper figure: top-down trajectories of the EB prompt and S2M on three EB-Navigation TEST episodes.

  python scripts/make_trajectory_figure.py     -> paper/latex/fig/trajectories.pdf (+ prints caption facts)

Reads only saved episode records (runs/eb_nav/test/...) and the scenes' reachable floor positions
(paper/figures/reachable_positions.json, from AI2-THOR GetReachablePositions at grid 0.1 m).
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_nav_failures import classify  # noqa: E402

TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
EB, S2M = "eb_Qwen3-VL-4B-Instruct-FP8_eb_official_r1", "s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1"
CAPTION = os.path.join(ROOT, "paper", "latex", "sec", "fig_traj.tex")


def pick(sub, want, prefer):
    """First test episode of `sub` whose (EB outcome, S2M outcome) satisfies `want`, ordered by `prefer`."""
    out = []
    for f in sorted(os.listdir(os.path.join(TEST, S2M, sub)), key=lambda x: int(x.split("_")[1].split(".")[0])):
        pe, ps = os.path.join(TEST, EB, sub, f), os.path.join(TEST, S2M, sub, f)
        if not os.path.exists(pe):
            continue
        eb, s2m = json.load(open(pe)), json.load(open(ps))
        if want(eb, s2m):
            out.append((prefer(eb, s2m), len(out), f))
    if not out:
        raise SystemExit(f"no episode in {sub} matches")
    return sorted(out)[0][2]


def hyp_dist(rec):
    t = rec["target"]
    return min((math.hypot(h[0] - t["x"], h[1] - t["z"]) for e in rec["perception_log"] for h in e["hyps"]), default=99)


FIRST = lambda e, s: 0  # take the first matching episode in index order (no selection on how striking it is)
PANELS = [
    ("long_horizon", lambda e, s: classify(e) == "never_seen" and s["task_success"] >= 1, FIRST,
     "(a) Long-horizon: target behind the agent"),
    ("base", lambda e, s: classify(e) == "blocked" and s["task_success"] >= 1, FIRST,
     "(b) Base: furniture blocks the EB prompt"),
    ("visual_appearance", lambda e, s: e["task_success"] >= 1 and s["task_success"] < 1
     and s["perception_log"][-1]["hyps"] and hyp_dist(s) > 1.5, FIRST,
     "(c) Visual: only look-alikes proposed"),
]
# reference palette (validated light mode): series 1 blue, series 2 orange; neutrals for everything else
C_S2M, C_EB = "#C55A11", "#7F7F7F"  # the colours of these agents in every figure of the paper
INK, INK2, FLOOR, WALL, SURFACE = "#0b0b0b", "#595959", "#EEEDEB", "#BFBFBF", "#ffffff"
OUT = os.path.join(ROOT, "paper", "latex", "fig", "trajectories.pdf")


def path(rec):
    return [t["x"] for t in rec["trajectory"]], [t["z"] for t in rec["trajectory"]]


OUTCOME = {"success": "success", "never_seen": "target never seen", "blocked": "blocked",
           "lost_after_seen": "target lost", "near_miss": "near miss", "approach_fail": "failed",
           "oscillation": "oscillates"}


def floor_image(pts, step=0.1):
    """Reachable positions on the 0.1 m grid as a filled occupancy image (1 = floor)."""
    xs, zs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    x0, z0 = xs.min() - step, zs.min() - step
    img = np.zeros((int(round((zs.max() - z0) / step)) + 2, int(round((xs.max() - x0) / step)) + 2))
    img[np.round((zs - z0) / step).astype(int), np.round((xs - x0) / step).astype(int)] = 1
    return img, (x0 - step / 2, x0 + (img.shape[1] - 0.5) * step, z0 - step / 2, z0 + (img.shape[0] - 0.5) * step)


def outcome(rec):
    c = classify(rec)
    return f"{OUTCOME.get(c, c)}, {rec['num_steps']} steps" if c == "success" else OUTCOME.get(c, c)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5})
    floors = json.load(open(os.path.join(ROOT, "paper", "figures", "reachable_positions.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    facts = []
    for ax, (sub, want, prefer, title) in zip(axes, PANELS):
        ep = pick(sub, want, prefer)
        eb = json.load(open(os.path.join(TEST, EB, sub, ep)))
        s2m = json.load(open(os.path.join(TEST, S2M, sub, ep)))
        img, ext = floor_image(floors[s2m["scene"]])
        ax.imshow(np.ma.masked_equal(img, 0), extent=ext, origin="lower", cmap=matplotlib.colors.ListedColormap([FLOOR]),
                  interpolation="nearest", zorder=0)
        gx = np.linspace(ext[0], ext[1], img.shape[1] + 1)[:-1] + (ext[1] - ext[0]) / img.shape[1] / 2
        gz = np.linspace(ext[2], ext[3], img.shape[0] + 1)[:-1] + (ext[3] - ext[2]) / img.shape[0] / 2
        ax.contour(gx, gz, img, levels=[0.5], colors=WALL, linewidths=0.7, zorder=1)
        for rec, col, ls, z in [(eb, C_EB, (0, (3.2, 1.6)), 2), (s2m, C_S2M, "-", 3)]:
            xs, zs = path(rec)
            ax.plot(xs, zs, color=col, lw=1.5, ls=ls, solid_joinstyle="round", solid_capstyle="round",
                    dash_capstyle="round", zorder=z)
            ax.scatter([xs[-1]], [zs[-1]], s=12, marker="o", facecolor=col, edgecolor=SURFACE, lw=0.5, zorder=z + 5)
            rej = {}
            for t in rec["trajectory"][1:]:
                if not t["ok"]:
                    k = (round(t["x"], 2), round(t["z"], 2))
                    rej[k] = rej.get(k, 0) + 1
            for (rx, rz), n in rej.items():
                ax.scatter([rx], [rz], s=20, marker="x", c=col, lw=1.0, zorder=z + 5)
                if n > 1:
                    ax.text(rx + 0.12, rz + 0.08, f"{n}$\\times$", color=INK2, fontsize=6.5, zorder=9)
        hyps = s2m["perception_log"][-1]["hyps"]
        for hx, hz, w, exhausted in hyps:
            size = 14 + 22 * min(max(w, 0.0), 2.0)
            ax.scatter([hx], [hz], s=size, marker="o", zorder=4, lw=1.0,
                       facecolor=SURFACE if exhausted else C_S2M, edgecolor=C_S2M if exhausted else SURFACE)
        t0 = s2m["trajectory"][0]
        y = math.radians(t0["yaw"])
        ax.annotate("", xy=(t0["x"] + 0.45 * math.sin(y), t0["z"] + 0.45 * math.cos(y)), xytext=(t0["x"], t0["z"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6), zorder=5)
        ax.scatter([t0["x"]], [t0["z"]], s=18, c=INK, zorder=6, lw=0)
        ax.scatter([s2m["target"]["x"]], [s2m["target"]["z"]], s=70, marker="*", c=INK, edgecolor=SURFACE,
                   lw=0.6, zorder=7)
        px = [ext[0], ext[1], s2m["target"]["x"]] + [h[0] for h in hyps]
        pz = [ext[2], ext[3], s2m["target"]["z"]] + [h[1] for h in hyps]
        x0, x1, z0, z1 = min(px) - 0.25, max(px) + 0.25, min(pz) - 0.25, max(pz) + 0.25
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("equal")
        ax.plot([x0 + 0.15, x0 + 1.15], [z0 + 0.12] * 2, color=INK2, lw=1.0, solid_capstyle="butt")
        ax.text(x0 + 0.65, z0 + 0.2, "1 m", color=INK2, ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, color=INK, loc="left", pad=19)
        ax.text(0.0, 1.075, "S2M: " + outcome(s2m), transform=ax.transAxes, color=C_S2M, fontsize=6.5,
                ha="left", va="bottom")
        ax.text(0.0, 1.01, "EB prompt: " + outcome(eb), transform=ax.transAxes, color=INK2, fontsize=6.5,
                ha="left", va="bottom")
        facts.append(dict(sub=sub, ep=ep, instruction=s2m["instruction"], eb=classify(eb), eb_steps=eb["num_steps"],
                          eb_rej=sum(not t["ok"] for t in eb["trajectory"][1:]), s2m=classify(s2m),
                          s2m_steps=s2m["num_steps"], nh=len(hyps)))
        print(f"{title}: scene {s2m['scene']}; EB {classify(eb)} ({eb['num_steps']} steps); "
              f"S2M {classify(s2m)} ({s2m['num_steps']} steps); hypotheses {len(hyps)}")
    handles = [Line2D([], [], color=C_S2M, lw=1.5, label="S2M"),
               Line2D([], [], color=C_EB, lw=1.5, ls=(0, (3.2, 1.6)), label="EB prompt"),
               Line2D([], [], color=INK, marker="o", ls="", ms=4, label="start"),
               Line2D([], [], color=INK, marker="*", ls="", ms=7, label="target"),
               Line2D([], [], color=INK2, marker="x", ls="", ms=4, label="rejected move"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=C_S2M, mec=SURFACE, label="hypothesis"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=SURFACE, mec=C_S2M, label="exhausted hypothesis")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7, handletextpad=0.35,
               columnspacing=1.1, labelcolor=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.09, wspace=0.06)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_paper_figures import check
    check(fig, "trajectories")
    fig.savefig(OUT)
    cap = ("\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{fig/trajectories.pdf}\n"
           "\\caption{Trajectories on three EB-Navigation test episodes (Qwen3-VL-4B), with the hypotheses of \\ours at "
           "the end of the episode sized by evidence. Grey: reachable floor.}\n"
           "\\label{fig:traj}\n\\end{figure*}\n")
    open(CAPTION, "w").write(cap)
    print("wrote", OUT, "and", CAPTION)


if __name__ == "__main__":
    main()
