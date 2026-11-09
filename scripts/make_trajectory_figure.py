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
C_S2M, C_EB = "#C55A11", "#4D4D4D"  # the colours of these agents in every figure of the paper
INK, INK2, FLOOR, WALL, SURFACE = "#0b0b0b", "#404040", "#EBEAE7", "#8C8C8C", "#ffffff"
THESIS = os.environ.get("THESIS") == "1"  # thesis variant: thesis text width, text scaled by K, *_thesis outputs
K = 1.25 if THESIS else 1.0
OUT = os.path.join(ROOT, "paper", "latex", "fig", "trajectories_thesis.pdf" if THESIS else "trajectories.pdf")


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
                         "mathtext.fontset": "stix", "font.size": 7 * K, "axes.titlesize": 7.5 * K})
    floors = json.load(open(os.path.join(ROOT, "paper", "figures", "reachable_positions.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.75) if THESIS else (6.9, 2.1))
    heads = []
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
        from scipy.ndimage import gaussian_filter
        ax.contour(gx, gz, gaussian_filter(img.astype(float), 0.8), levels=[0.5], colors=WALL, linewidths=0.8, zorder=1)
        for rec, col, ls, z in [(eb, C_EB, (0, (4.0, 1.8)), 2), (s2m, C_S2M, "-", 3)]:
            xs, zs = path(rec)
            ax.plot(xs, zs, color=col, lw=1.5, ls=ls, solid_joinstyle="round", solid_capstyle="round",
                    dash_capstyle="round", zorder=z)
            ax.scatter([xs[-1]], [zs[-1]], s=20, marker="o", c=col, lw=0, zorder=z + 6)
            rej = []  # [x, z, count]: rejections within 0.45 m of each other are one place
            for t in rec["trajectory"][1:]:
                if not t["ok"]:
                    c = next((c for c in rej if math.hypot(c[0] - t["x"], c[1] - t["z"]) < 0.45), None)
                    if c:
                        c[2] += 1
                    else:
                        rej.append([t["x"], t["z"], 1])
            for rx, rz, n in rej:
                ax.scatter([rx], [rz], s=22, marker="x", c=col, lw=1.1, zorder=z + 5)
                if n > 1:  # count beside the mark, on the side toward the middle of the room
                    # count above the mark, on the side toward the middle of the room, clear of the start marker
                    dx = 0.25 if rx < (ext[0] + ext[1]) / 2 else -0.25
                    ax.text(rx + dx, rz + 0.3, f"{n}$\\times$", color=INK, fontsize=6.8 * K, zorder=9, va="bottom",
                            ha="left" if dx > 0 else "right")
        hyps = s2m["perception_log"][-1]["hyps"]
        for hx, hz, w, exhausted in hyps:
            ax.scatter([hx], [hz], s=125, marker="o", zorder=8, lw=1.1,
                       facecolor="none", edgecolor=C_EB if exhausted else C_S2M)
        t0 = s2m["trajectory"][0]
        y = math.radians(t0["yaw"])
        ax.scatter([t0["x"]], [t0["z"]], s=42, marker=(3, 0, -math.degrees(y)), c=INK, zorder=6, lw=0)
        ax.scatter([s2m["target"]["x"]], [s2m["target"]["z"]], s=46, marker="*", c=INK, lw=0, zorder=7)
        px = path(eb)[0] + path(s2m)[0] + [s2m["target"]["x"]] + [h[0] for h in hyps]
        pz = path(eb)[1] + path(s2m)[1] + [s2m["target"]["z"]] + [h[1] for h in hyps]
        x0, x1 = min(ext[0], min(px) - 0.3) - 0.05, max(ext[1], max(px) + 0.3) + 0.05
        z0, z1 = min(ext[2], min(pz) - 0.3) - 0.32, max(ext[3], max(pz) + 0.3) + 0.05
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("equal")
        ax.plot([x1 - 1.2, x1 - 0.2], [z0 + 0.06] * 2, color=INK, lw=1.1, solid_capstyle="butt", zorder=9)
        ax.text(x1 - 1.27, z0 + 0.06, "1 m", color=INK, ha="right", va="center", fontsize=6.6 * K, zorder=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # thesis: the descriptive part of each title is in the caption, so the short title fits the narrower panel
        heads.append((ax, title.split(":")[0] if THESIS else title, "S2M: " + outcome(s2m), "EB prompt: " + outcome(eb)))
        facts.append(dict(sub=sub, ep=ep, instruction=s2m["instruction"], eb=classify(eb), eb_steps=eb["num_steps"],
                          eb_rej=sum(not t["ok"] for t in eb["trajectory"][1:]), s2m=classify(s2m),
                          s2m_steps=s2m["num_steps"], nh=len(hyps)))
        print(f"{title}: scene {s2m['scene']}; EB {classify(eb)} ({eb['num_steps']} steps); "
              f"S2M {classify(s2m)} ({s2m['num_steps']} steps); hypotheses {len(hyps)}")
    handles = [Line2D([], [], color=C_S2M, lw=1.5, label="S2M"),
               Line2D([], [], color=C_EB, lw=1.5, ls=(0, (3.2, 1.6)), label="EB prompt"),
               Line2D([], [], color=INK, marker=">", ls="", ms=5, label="start and heading"),
               Line2D([], [], color=INK, marker="*", ls="", ms=7, label="target"),
               Line2D([], [], color=INK2, marker="x", ls="", ms=4.5, mew=1.1, label="rejected moves ($n\\times$)"),
               (Line2D([], [], color=C_S2M, marker="o", ls="", ms=4), Line2D([], [], color=C_EB, marker="o", ls="", ms=4)),
               Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=C_S2M, mew=1.3, label="S2M hypothesis at the end")]
    from matplotlib.legend_handler import HandlerTuple
    labels = [h.get_label() if not isinstance(h, tuple) else "end of path" for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=4 if THESIS else 7, frameon=False, fontsize=7 * K, handletextpad=0.35,
               columnspacing=1.0, labelcolor=INK, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    fig.subplots_adjust(left=0.003, right=0.997, top=0.76 if THESIS else 0.77, bottom=0.2 if THESIS else 0.15, wspace=0.03)
    # panel headers at one height for all panels (panels differ in height because each map keeps its aspect)
    fig.canvas.draw()
    top = max(a.get_position().y1 for a, *_ in heads)
    lh = 9.0 * K / 72 / fig.get_figheight()  # one text line in figure units
    for a, t, l1, l2 in heads:
        x0 = a.get_position().x0
        fig.text(x0, top + 0.012 + 2 * lh, t, color=INK, fontsize=7.5 * K, ha="left", va="bottom")
        fig.text(x0, top + 0.012 + lh, l1, color=C_S2M, fontsize=6.8 * K, ha="left", va="bottom")
        fig.text(x0, top + 0.012, l2, color=C_EB, fontsize=6.8 * K, ha="left", va="bottom")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_paper_figures import check
    check(fig, "trajectories")
    fig.savefig(OUT)
    cap = ("\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{fig/trajectories.pdf}\n"
           "\\caption{Trajectories on three EB-Navigation test episodes (Qwen3-VL-4B) and the hypotheses of \\ours at the end "
           "of each. Grey: reachable floor; targets on furniture lie outside it. $n\\times$: rejected moves at one spot.}\n"
           "\\label{fig:traj}\n\\end{figure*}\n")
    if not THESIS:  # the thesis keeps its own copy of the caption file
        open(CAPTION, "w").write(cap)
    print("wrote", OUT, "and", CAPTION)


if __name__ == "__main__":
    main()
