"""Paper figure: top-down trajectories of the EB prompt and S2M on three EB-Navigation TEST episodes.

  python scripts/make_trajectory_figure.py     -> paper/latex/fig/trajectories.pdf (+ prints caption facts)

Reads only saved episode records (runs/eb_nav/test/...) and the scenes' reachable floor positions
(paper/figures/reachable_positions.json, from AI2-THOR GetReachablePositions at grid 0.1 m).
Target positions are drawn for illustration only; neither agent receives them.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_nav_failures import classify  # noqa: E402

TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
EB, S2M = "eb_Qwen3-VL-4B-Instruct-AWQ_eb_official_r1", "s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1"
PANELS = [("long_horizon", "episode_14.json", "(a) Long-horizon: the target starts behind the agent"),
          ("base", "episode_19.json", "(b) Base: an obstacle blocks the direct route"),
          ("visual_appearance", "episode_28.json", "(c) Visual: the target is never proposed")]
# reference palette (validated light mode): series 1 blue, series 2 orange; neutrals for everything else
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, FLOOR, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#ffffff"
OUT = os.path.join(ROOT, "paper", "latex", "fig", "trajectories.pdf")


def path(rec):
    return [t["x"] for t in rec["trajectory"]], [t["z"] for t in rec["trajectory"]]


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7})
    floors = json.load(open(os.path.join(ROOT, "paper", "figures", "reachable_positions.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45))
    for ax, (sub, ep, title) in zip(axes, PANELS):
        eb = json.load(open(os.path.join(TEST, EB, sub, ep)))
        s2m = json.load(open(os.path.join(TEST, S2M, sub, ep)))
        fx, fz = zip(*floors[s2m["scene"]])
        ax.scatter(fx, fz, s=2.2, marker="s", c=FLOOR, lw=0, zorder=0)
        for rec, col, z in [(eb, ORANGE, 2), (s2m, BLUE, 3)]:
            xs, zs = path(rec)
            ax.plot(xs, zs, color=col, lw=1.3, solid_joinstyle="round", solid_capstyle="round", zorder=z)
            # a move the simulator rejected leaves the agent in place: mark where it happened, with a count
            rej = {}
            for t in rec["trajectory"][1:]:
                if not t["ok"]:
                    k = (round(t["x"], 2), round(t["z"], 2))
                    rej[k] = rej.get(k, 0) + 1
            for (rx, rz), n in rej.items():
                ax.scatter([rx], [rz], s=22, marker="x", c=col, lw=1.0, zorder=z + 5)
                if n > 1:
                    ax.text(rx + 0.12, rz + 0.08, f"{n}$\\times$", color=INK2, fontsize=6, zorder=9)
        hyps = s2m["perception_log"][-1]["hyps"]
        for hx, hz, w, exhausted in hyps:
            size = 14 + 22 * min(max(w, 0.0), 2.0)
            ax.scatter([hx], [hz], s=size, marker="o", zorder=4, lw=1.0,
                       facecolor=SURFACE if exhausted else BLUE, edgecolor=BLUE if exhausted else SURFACE)
        t0 = s2m["trajectory"][0]
        y = math.radians(t0["yaw"])
        ax.annotate("", xy=(t0["x"] + 0.45 * math.sin(y), t0["z"] + 0.45 * math.cos(y)), xytext=(t0["x"], t0["z"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6), zorder=5)
        ax.scatter([t0["x"]], [t0["z"]], s=16, c=INK, zorder=6, lw=0)
        ax.scatter([s2m["target"]["x"]], [s2m["target"]["z"]], s=60, marker="*", c=INK, edgecolor=SURFACE,
                   lw=0.6, zorder=7)
        # objects stand on furniture outside the reachable floor: include target and hypotheses in the extent
        px = list(fx) + [s2m["target"]["x"]] + [h[0] for h in hyps]
        pz = list(fz) + [s2m["target"]["z"]] + [h[1] for h in hyps]
        x0, x1, z0, z1 = min(px) - 0.3, max(px) + 0.3, min(pz) - 0.3, max(pz) + 0.3
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("equal")
        ax.plot([x0 + 0.15, x0 + 1.15], [z0 + 0.15] * 2, color=INK2, lw=1.0)
        ax.text(x0 + 0.65, z0 + 0.22, "1 m", color=INK2, ha="center", va="bottom", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, color=INK, loc="left", pad=3)
        print(f"{title}: scene {s2m['scene']}; instruction {s2m['instruction']!r}; "
              f"EB {classify(eb)} ({eb['num_steps']} steps, final {eb['final_distance']:.2f} m); "
              f"S2M {classify(s2m)} ({s2m['num_steps']} steps, final {s2m['final_distance']:.2f} m); "
              f"S2M hypotheses {len(hyps)}, exhausted {sum(h[3] for h in hyps)}")
    handles = [Line2D([], [], color=BLUE, lw=1.3, label="\\ours path".replace("\\ours", "S2M")),
               Line2D([], [], color=ORANGE, lw=1.3, label="EB prompt path"),
               Line2D([], [], color=INK2, marker="x", ls="", ms=4, label="rejected move"),
               Line2D([], [], color=INK, marker="o", ls="", ms=4, label="start (arrow: heading)"),
               Line2D([], [], color=INK, marker="*", ls="", ms=7, label="target (not given to agents)"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=BLUE, mec=SURFACE, label="S2M hypothesis"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=SURFACE, mec=BLUE, label="exhausted hypothesis"),
               Line2D([], [], marker="s", ls="", ms=4, mfc=FLOOR, mec=FLOOR, label="reachable floor")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=6, handletextpad=0.35,
               columnspacing=0.8, labelcolor=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.1, wspace=0.04)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
