"""Replay a saved EB-Navigation episode in AI2-THOR and record it as an MP4 (no model needed).

The episode is reset in the simulator and the recorded actions are executed again; every pose is checked
against the record. What the VLM answered and what S2M decided at each step come from the saved episode record
(the same runs as the paper), and S2M's occupancy map is rebuilt from the replayed depth exactly as the agent
built it. The true target (star) is drawn for the viewer only.

  python demo/replay.py --subset long_horizon --episode 14 --agent both
  python demo/replay.py --subset base --episode 19 --agent s2m --lang en
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import viz  # noqa: E402

RUNS = {"s2m": "s2m_{model}_final_r1", "eb": "eb_{model}_eb_official_r1"}
MODELS = {"qwen3": "Qwen3-VL-4B-Instruct-AWQ", "qwen25": "Qwen2.5-VL-3B-Instruct-AWQ"}
SUBSETS = ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"]


def make_env(subset):
    from embagent.eb import shim
    shim.install()
    from embodiedbench.envs.eb_navigation import EBNavEnv
    env = EBNavEnv.EBNavigationEnv(eval_set=subset, exp_name="demo", resolution=500, selected_indexes=[0])
    env.log_path = os.path.join(tempfile.gettempdir(), "s2m_demo_eb_logs")
    return env, env._load_dataset(subset), EBNavEnv.DISCRETE_SKILLSET


def reset(env, full, idx):
    env.dataset, env.selected_indexes = [full[idx]], [idx]
    env.number_of_episodes, env._current_episode_num = 1, 0
    env.reset()
    return env.env.last_event


def pose_of(ev):
    a = ev.metadata["agent"]
    return {"x": a["position"]["x"], "z": a["position"]["z"], "yaw": a["rotation"]["y"], "horizon": a["cameraHorizon"]}


def s2m_text(T, lang, spec, entry, plan, next_action, hyps):
    vlm = [(T["phrase"], spec.get("query", ""), 2)]
    if entry is None:
        return vlm, [(T["mode"], "—", 1)]
    vlm.append((T["visible"], f"P(yes) = {entry['p_vis']:.2f}", 1))
    vlm.append((T["where"], (T["box"] + " " + str(entry["box"])) if entry.get("box") else T["nobox"], 1))
    if entry.get("p_zoom") is not None:
        vlm.append((T["crop"], f"P(yes) = {entry['p_zoom']:.2f}", 1))
    if entry.get("tiled"):
        vlm.append((T["tiles"], T["tiles_v"].format(entry["tiled"]), 1))
    r = (plan or {}).get("reasoning", {})
    goal = r.get("target") or r.get("goal")
    live = [h for h in hyps if not h[3]]
    dec = [(T["mode"], T["modes"].get(r.get("mode", ""), r.get("mode", "")), 1),
           (T["goal"], f"({goal[0]:.2f}, {goal[1]:.2f})" if goal else T["none"], 1),
           (T["hyps"], T["hyps_v"].format(len(hyps), sum(h[2] >= 0.8 for h in live), sum(h[3] for h in hyps)), 1),
           (T["action"], T["actions"][next_action] if next_action is not None else T["none"], 1)]
    return vlm, dec


def eb_text(T, plan, next_action, last_ok):
    r = (plan or {}).get("reasoning", {})
    acts = r.get("executable_plan") or []
    names = ", ".join(T["actions"][a["action_id"]] for a in acts if isinstance(a, dict) and
                      isinstance(a.get("action_id"), int) and 0 <= a["action_id"] < 8)
    vlm = [(T["desc"], r.get("visual_state_description", "—"), 4),
           (T["reason"], r.get("reasoning_and_reflection", "—"), 5)]
    dec = [(T["plan"], names or T["none"], 3),
           (T["action"], T["actions"][next_action] if next_action is not None else T["none"], 1)]
    if last_ok is not None:
        dec.append((T["feedback"], T["ok"] if last_ok else T["rejected"], 1))
    return vlm, dec


def frames_for(agent, rec, env, full, idx, view, args, model_name):
    """Replay one agent's recorded actions; returns [(PIL frame, hold_frames)] and the number of pose mismatches."""
    from embagent.planners.geo import GeoCore
    T = viz.TEXT[args.lang]
    traj, plans = rec["trajectory"], rec["plans"]
    n = len(traj) - 1
    ev = reset(env, full, idx)
    core = GeoCore() if agent == "s2m" else None
    tgt = (rec["target"]["x"], rec["target"]["z"]) if args.show_target else None
    hold = max(1, round(args.hold * args.fps))
    out, mism, prev = [], 0, None
    for k in range(n + 1):
        if k > 0:
            env.step(traj[k]["action"], {}, 1 if k == 1 else 0)
            ev = env.env.last_event
        p = pose_of(ev)
        t = traj[k]
        mism += abs(p["x"] - t["x"]) + abs(p["z"] - t["z"]) > 1e-3 or abs(((p["yaw"] - t["yaw"] + 180) % 360) - 180) > 0.5
        if core is not None:
            if core.map is None:
                core.reset(p)
            if k > 0:
                core.record_move(prev, traj[k]["action"], bool(traj[k]["ok"]))
            core.ensure({"pose": p, "depth": ev.depth_frame, "step": k})
        prev = p
        plan = next((q for q in reversed(plans) if q["step"] <= k), None) if agent == "eb" else \
            (plans[k] if k < len(plans) else None)
        plog = rec.get("perception_log", [])
        entry = plog[k] if agent == "s2m" and k < len(plog) else None
        hyps = [tuple(h) for h in entry["hyps"]] if entry else []
        nxt = traj[k + 1]["action"] if k < n else None
        if agent == "s2m":
            vlm, dec = s2m_text(T, args.lang, rec.get("spec", {}), entry, plan, nxt, hyps)
            r = (plan or {}).get("reasoning", {})
            goal = r.get("target") or r.get("goal")
        else:
            vlm, dec = eb_text(T, plan, nxt, bool(t["ok"]) if k > 0 else None)
            goal = None
        st = {"agent": agent, "lang": args.lang, "model": model_name, "instruction": rec["instruction"],
              "step": min(k + 1, n), "n_steps": n, "rgb": ev.frame, "box": entry.get("box") if entry else None,
              "occ_map": core.map if core else None, "traj": [(q["x"], q["z"]) for q in traj[:k + 1]], "pose": t,
              "hyps": hyps, "goal": tuple(goal) if goal and k < n else None, "target": tgt, "vlm": vlm, "dec": dec,
              "timeline": [("ok" if traj[i]["ok"] else "rej") if i <= k else "todo" for i in range(1, n + 1)],
              "cur": k if k < n else -1, "rejected": k > 0 and not t["ok"],
              "outcome": (rec["task_success"] >= 1, rec["final_distance"]) if k == n else None,
              "note": args.note}
        out.append((viz.render(view, st), hold * (3 if k == n else 2 if k == 0 else 1)))
    return out, mism


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", default="long_horizon", choices=SUBSETS)
    ap.add_argument("--episode", type=int, default=14, help="episode number as in episode_<N>.json (1-based)")
    ap.add_argument("--agent", default="both", choices=["both", "s2m", "eb"])
    ap.add_argument("--model", default="qwen3", choices=list(MODELS))
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--hold", type=float, default=0.8, help="seconds per step")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-target", dest="show_target", action="store_false", help="hide the true target star")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model = MODELS[args.model]
    T = viz.TEXT[args.lang]
    args.note = ("回放：动作、模型回答和决策均来自论文测试集的保存记录；星号是真实目标，仅供观众参考。" if args.lang == "zh" else
                 "Replay: actions, model answers and decisions come from the saved test-set records; the star is "
                 "the true target, shown to the viewer only.")
    agents = ["eb", "s2m"] if args.agent == "both" else [args.agent]
    name = f"episode_{args.episode}.json"
    recs = {a: json.load(open(os.path.join(ROOT, "runs", "eb_nav", "test", RUNS[a].format(model=model), args.subset, name)))
            for a in agents}
    env, full, _ = make_env(args.subset)
    idx = args.episode - 1
    reset(env, full, idx)
    floor = [(q["x"], q["z"]) for q in env.env.step(action="GetReachablePositions").metadata["actionReturn"]]
    extra = [(t["x"], t["z"]) for r in recs.values() for t in r["trajectory"]]
    extra += [tuple(h[:2]) for r in recs.values() for e in r.get("perception_log", []) for h in e["hyps"]]
    extra.append((recs[agents[0]]["target"]["x"], recs[agents[0]]["target"]["z"]))
    view = viz.MapView(floor, extra)
    seqs = {}
    for a in agents:
        seqs[a], mism = frames_for(a, recs[a], env, full, idx, view, args, model)
        r = recs[a]
        print(f"{a}: {len(r['trajectory']) - 1} steps, success={r['task_success']:.0f}, final distance "
              f"{r['final_distance']:.2f} m; replayed poses that differ from the record: {mism}")
    env.env.stop()
    out = args.out or os.path.join(HERE, "videos", f"{args.subset}_ep{args.episode}_{args.agent}_{args.model}_{args.lang}.mp4")
    size = (viz.W * len(agents), viz.H)
    vw = viz.VideoWriter(out, size, fps=args.fps)
    sub = {"zh": "左：智能体第一视角；中：俯视地图；右：小模型的回答与控制器的决策。",
           "en": "Left: the agent's view. Middle: top-down map. Right: the VLM's answers and the controller's decisions."}
    head = (" vs. ".join(T[a] for a in agents), "“" + recs[agents[0]]["instruction"] + "”")
    vw.write(viz.title_card(head, sub[args.lang] + " " + args.note, args.lang, size), repeat=3 * args.fps)
    if len(agents) == 2:
        a, b = seqs["eb"], seqs["s2m"]
        L = max(len(a), len(b))
        a += [(a[-1][0], a[-1][1])] * (L - len(a))
        b += [(b[-1][0], b[-1][1])] * (L - len(b))
        for (fa, ha), (fb, hb) in zip(a, b):
            vw.write(viz.side_by_side(fa, fb), repeat=max(ha, hb))
    else:
        for f, h in seqs[agents[0]]:
            vw.write(f, repeat=h)
    vw.close()
    print(f"wrote {out} ({vw.frames} frames, {vw.frames / args.fps:.1f} s, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
