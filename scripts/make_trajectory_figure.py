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
INK, INK2, FLOOR, SURFACE = "#0b0b0b", "#595959", "#E7E6E6", "#ffffff"
OUT = os.path.join(ROOT, "paper", "latex", "fig", "trajectories.pdf")


def path(rec):
    return [t["x"] for t in rec["trajectory"]], [t["z"] for t in rec["trajectory"]]


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5})
    floors = json.load(open(os.path.join(ROOT, "paper", "figures", "reachable_positions.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    facts = []
    for ax, (sub, want, prefer, title) in zip(axes, PANELS):
        ep = pick(sub, want, prefer)
        eb = json.load(open(os.path.join(TEST, EB, sub, ep)))
        s2m = json.load(open(os.path.join(TEST, S2M, sub, ep)))
        fx, fz = zip(*floors[s2m["scene"]])
        ax.scatter(fx, fz, s=2.2, marker="s", c=FLOOR, lw=0, zorder=0)
        for rec, col, z in [(eb, C_EB, 2), (s2m, C_S2M, 3)]:
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
                    ax.text(rx + 0.12, rz + 0.08, f"{n}$\\times$", color=INK2, fontsize=6.6, zorder=9)
        hyps = s2m["perception_log"][-1]["hyps"]
        for hx, hz, w, exhausted in hyps:
            size = 14 + 22 * min(max(w, 0.0), 2.0)
            ax.scatter([hx], [hz], s=size, marker="o", zorder=4, lw=1.0,
                       facecolor=SURFACE if exhausted else C_S2M, edgecolor=C_S2M if exhausted else SURFACE)
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
        ax.text(x0 + 0.65, z0 + 0.22, "1 m", color=INK2, ha="center", va="bottom", fontsize=6.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, color=INK, loc="left", pad=3)
        facts.append(dict(sub=sub, ep=ep, instruction=s2m["instruction"], eb=classify(eb), eb_steps=eb["num_steps"],
                          eb_rej=sum(not t["ok"] for t in eb["trajectory"][1:]), s2m=classify(s2m),
                          s2m_steps=s2m["num_steps"], s2m_rej=sum(not t["ok"] for t in s2m["trajectory"][1:]),
                          nh=len(hyps), hmin=min(math.hypot(h[0] - s2m["target"]["x"], h[1] - s2m["target"]["z"]) for h in hyps) if hyps else None,
                          hmax=max(math.hypot(h[0] - s2m["target"]["x"], h[1] - s2m["target"]["z"]) for h in hyps) if hyps else None,
                          turns=sum(t["action"] in (4, 5) for t in s2m["trajectory"][1:5])))
        print(f"{title}: scene {s2m['scene']}; instruction {s2m['instruction']!r}; "
              f"EB {classify(eb)} ({eb['num_steps']} steps, final {eb['final_distance']:.2f} m); "
              f"S2M {classify(s2m)} ({s2m['num_steps']} steps, final {s2m['final_distance']:.2f} m); "
              f"S2M hypotheses {len(hyps)}, exhausted {sum(h[3] for h in hyps)}")
    handles = [Line2D([], [], color=C_S2M, lw=1.3, label="S2M path"),
               Line2D([], [], color=C_EB, lw=1.3, label="EB prompt path"),
               Line2D([], [], color=INK2, marker="x", ls="", ms=4, label="rejected move"),
               Line2D([], [], color=INK, marker="o", ls="", ms=4, label="start (arrow: heading)"),
               Line2D([], [], color=INK, marker="*", ls="", ms=7, label="target (not given to agents)"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=C_S2M, mec=SURFACE, label="S2M hypothesis"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=SURFACE, mec=C_S2M, label="exhausted hypothesis"),
               Line2D([], [], marker="s", ls="", ms=4, mfc=FLOOR, mec=FLOOR, label="reachable floor")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7, handletextpad=0.35,
               columnspacing=0.8, labelcolor=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.18, wspace=0.04)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_paper_figures import check
    check(fig, "trajectories")
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    a, b, c = facts
    look = c["instruction"].split("navigate to the ")[-1].split(" in the room")[0] if "navigate to" in c["instruction"] else c["instruction"]
    cap = (f"\\begin{{figure*}}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{{fig/trajectories.pdf}}\n"
           f"\\caption{{Top-down trajectories of the EB prompt (grey) and \\ours (orange) with Qwen3-VL-4B on three test "
           f"episodes; dots are the hypotheses of \\ours at the end of the episode, sized by evidence. "
           f"(a)~The EB prompt never brings the target into view in {a['eb_steps']} steps; \\ours turns, "
           f"commits to a hypothesis {a['hmin']:.1f}\\,m from the target, and succeeds in {a['s2m_steps']} steps. "
           f"(b)~{b['eb_rej']} of the {b['eb_steps']} moves of the EB prompt are rejected; \\ours plans around the furniture "
           f"on its occupancy map and succeeds in {b['s2m_steps']} steps. (c)~A Visual failure of \\ours (``{look}''): its "
           + (f"hypothesis lies {c['hmin']:.1f}\\,m" if c['nh'] == 1 else f"{c['nh']} hypotheses lie {c['hmin']:.1f}--{c['hmax']:.1f}\\,m")
           + f" from the target, while the EB prompt succeeds.}}\n"
           f"\\label{{fig:traj}}\n\\end{{figure*}}\n")
    open(CAPTION, "w").write("% generated by scripts/make_trajectory_figure.py -- do not edit\n" + cap)
    print("wrote", OUT, "and", CAPTION)


if __name__ == "__main__":
    main()
