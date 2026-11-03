"""Side-by-side replay video of one saved EB-Navigation test episode: EB prompt (left) vs S2M (right).

Both episodes are reset in AI2-THOR and their recorded actions are executed again; every executed pose is checked
against the record. The model's answers and the controller's decisions come from the saved records of the paper's
runs, and S2M's occupancy map is rebuilt from the replayed depth. Motion between steps is rendered by a display
camera. No model is needed. Thinking time is not reproduced: every decision is held for --pause seconds.

  python demo/replay.py --subset long_horizon --episode 14
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import render as R  # noqa: E402
from agents import EBRecord, S2MRecord  # noqa: E402
from sim import SUBSETS, AgentSim, episode, pose_of  # noqa: E402

MODELS = {"qwen3": "Qwen3-VL-4B-Instruct-AWQ", "qwen25": "Qwen2.5-VL-3B-Instruct-AWQ"}
RUNS = {"s2m": "s2m_{m}_final_r1", "eb": "eb_{m}_eb_official_r1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", default="long_horizon", choices=SUBSETS)
    ap.add_argument("--episode", type=int, default=14, help="episode number as in episode_<N>.json (1-60)")
    ap.add_argument("--model", default="qwen3", choices=list(MODELS))
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds each decision stays on screen")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model, T = MODELS[args.model], R.TEXT[args.lang]
    recs = {a: json.load(open(os.path.join(ROOT, "runs", "eb_nav", "test", RUNS[a].format(m=model), args.subset,
                                           f"episode_{args.episode}.json"))) for a in R.AGENTS}
    sims, starts = {}, {}
    for a in R.AGENTS:
        sims[a] = AgentSim()
        sims[a].load("test", args.subset)
        starts[a] = sims[a].reset(args.episode - 1)
    _, floor, target, instr = starts["s2m"]
    extra = [(t["x"], t["z"]) for r in recs.values() for t in r["trajectory"]] + [target]
    extra += [tuple(h[:2]) for e in recs["s2m"].get("perception_log", []) for h in e["hyps"]]
    view = R.MapView(floor, extra)
    gens = {}
    for a in R.AGENTS:
        src = S2MRecord(recs[a]) if a == "s2m" else EBRecord(recs[a])
        panel = R.Panel(a, args.lang, model, view, sims[a].budget, target)
        gens[a] = episode(sims[a], src, panel, starts[a][0], instr, args.fps, pause=args.pause)
    head = R.header(args.lang, "replay", "test", args.subset, args.episode, instr)
    out = args.out or os.path.join(HERE, "videos", f"compare_{args.subset}_ep{args.episode}_{args.model}_{args.lang}.mp4")
    vw = R.VideoWriter(out, fps=args.fps)
    vw.write(R.title_card((T["title"], "“" + instr + "”"), T["note_replay"]), repeat=3 * args.fps)
    cur, until, done, results, mism = {}, {a: 0.0 for a in R.AGENTS}, {a: False for a in R.AGENTS}, {}, {a: 0 for a in R.AGENTS}

    def advance(a):
        while True:
            try:
                item = next(gens[a])
            except StopIteration:
                done[a] = True
                return
            if item[0] == "frame":
                cur[a], until[a] = item[1], until[a] + item[2]
                return
            if item[0] == "step":
                s, p = item[1]["step"], pose_of(sims[a].c.last_event)
                r = recs[a]["trajectory"][s]
                mism[a] += (abs(p["x"] - r["x"]) + abs(p["z"] - r["z"]) > 1e-3
                            or abs((p["yaw"] - r["yaw"] + 180) % 360 - 180) > .5)
            elif item[0] == "done":
                results[a] = item[1]

    for a in R.AGENTS:
        advance(a)
    t, dt = 0.0, 1.0 / args.fps
    while not all(done.values()):
        for a in R.AGENTS:
            while not done[a] and until[a] <= t + 1e-9:
                advance(a)
        vw.write(R.compose(head, cur["eb"], cur["s2m"]))
        t += dt
    vw.close()
    for a in R.AGENTS:
        sims[a].close()
        r, rec = results[a], recs[a]
        print(f"{a}: {r['steps']} steps, success={r['success']} (record: {rec['task_success'] >= 1}), "
              f"distance {r['distance']} m; replayed poses that differ from the record: {mism[a]}")
    print(f"wrote {out} ({vw.frames} frames, {vw.frames / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
