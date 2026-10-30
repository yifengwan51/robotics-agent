"""Run S2M live on one EB-Navigation episode and record what it sees and decides as an MP4.

Needs the VLM server (bash demo/run_demo.sh serve). Uses the frozen S2M configuration of the paper. A live run
can differ slightly from the saved record of the same episode, because batched GPU inference is not bit-exact;
paper numbers come from the saved records (see replay.py). For a browser view while it runs, use live_view.py.

  python demo/live.py --subset base --episode 19
  python demo/live.py --split dev --subset visual_appearance --episode 3 --preview demo/videos/preview.png
"""
import argparse
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import viz  # noqa: E402
from replay import MODELS, SUBSETS, make_env, pose_of, reset, s2m_text  # noqa: E402

NOTE = {"zh": "实时运行：S2M 正在调用本地小模型；星号是真实目标，仅供观众参考，智能体看不到。",
        "en": "Live run: S2M is querying the local VLM; the star is the true target, shown to the viewer only."}


def check_server(model):
    url = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        served = urllib.request.urlopen(url + "/models", timeout=5).read().decode()
    except Exception:
        return f"No VLM server at {url}. Start it first: bash demo/run_demo.sh serve"
    if model not in served:
        return f"The server at {url} does not serve {model}. Restart it: bash demo/run_demo.sh serve"
    return None


def load_episodes(env, split, subset):
    """Point the EB environment at the test or dev episodes of one subset and return them."""
    from embodiedbench.envs.eb_navigation import EBNavEnv
    env.data_path = (os.path.join(ROOT, "data", "eb_nav_dev", f"{subset}.json") if split == "dev" else
                     os.path.join(os.path.dirname(EBNavEnv.__file__), "datasets", f"{subset}.json"))
    return env._load_dataset(subset)


def run_live(env, full, idx, planner, skills, model, lang="zh", fps=10, hold_s=0.8, out=None, delay=0.0,
             on_frame=None, on_step=None, should_stop=None):
    """Run S2M on episode `idx` of `full`, render every step into the MP4 `out`, and return a result dict.
    on_frame(img) receives every rendered frame, on_step(dict) every executed action; should_stop() aborts."""
    T = viz.TEXT[lang]
    reset(env, full, idx)
    floor = [(q["x"], q["z"]) for q in env.env.step(action="GetReachablePositions").metadata["actionReturn"]]
    tp = full[idx]["target_position"]
    view = viz.MapView(floor, [(tp["x"], tp["z"])], margin=0.8)
    ev = reset(env, full, idx)
    instr = env.episode_language_instruction
    planner.reset(instr)
    hold = max(1, round(hold_s * fps))
    vw = viz.VideoWriter(out, (viz.W, viz.H), fps=fps)

    def emit(img, rep):
        vw.write(img, repeat=rep)
        if on_frame:
            on_frame(img)

    emit(viz.title_card((T["s2m"], "“" + instr + "”"), NOTE[lang], lang), 3 * fps)
    traj, status, done, info, step, max_steps = [], [], False, {}, 0, env._max_episode_steps

    def frame(p, entry, plan, nxt, outcome=None):
        hyps = [tuple(h) for h in entry["hyps"]] if entry else []
        vlm, dec = s2m_text(T, lang, planner.spec or {}, entry, {"reasoning": plan or {}}, nxt, hyps)
        goal = (plan or {}).get("target") or (plan or {}).get("goal")
        return viz.render(view, {
            "agent": "s2m", "lang": lang, "model": model + " (live)", "instruction": instr,
            "step": min(step + 1, max_steps), "n_steps": max_steps, "rgb": ev.frame,
            "box": entry.get("box") if entry else None, "occ_map": planner.core.map, "traj": list(traj), "pose": p,
            "hyps": hyps, "goal": tuple(goal) if goal and outcome is None else None, "target": (tp["x"], tp["z"]),
            "vlm": vlm, "dec": dec, "timeline": status + ["todo"] * (max_steps - len(status)),
            "cur": -1 if outcome else len(status), "rejected": bool(status) and status[-1] == "rej",
            "outcome": outcome, "note": NOTE[lang]})

    p = pose_of(ev)
    traj.append((p["x"], p["z"]))
    aborted = False
    while not done:
        if should_stop and should_stop():
            aborted = True
            break
        obs = {"rgb": ev.frame, "depth": ev.depth_frame, "pose": p, "instruction": instr, "step": step}
        actions, reasoning = planner.act(obs)
        a = int(actions[0])
        emit(frame(p, planner.log[-1] if planner.log else None, reasoning, a), 2 * hold if step == 0 else hold)
        if delay:
            time.sleep(delay)
        _, _, done, info = env.step(a, reasoning, 1 if step == 0 else 0)
        ev = env.env.last_event
        p = pose_of(ev)
        step = info["env_step"]
        planner.observe(info, {"rgb": ev.frame, "depth": ev.depth_frame, "pose": p, "instruction": instr, "step": step})
        traj.append((p["x"], p["z"]))
        ok_move = bool(info["last_action_success"])
        status.append("ok" if ok_move else "rej")
        if on_step:
            on_step({"step": step, "action": skills[a], "action_id": a, "ok": ok_move, "mode": reasoning.get("mode"),
                     "distance": round(float(info["distance"]), 2)})
    ok = bool(info.get("task_success", 0) >= 1)
    dist = float(info.get("distance", float("nan"))) if info else float("nan")
    last = frame(p, planner.log[-1] if planner.log else None, None, None, outcome=None if aborted else (ok, dist))
    emit(last, 3 * hold)
    vw.close()
    return {"success": ok, "distance": round(dist, 2), "steps": step, "vlm_calls": planner.g.calls,
            "aborted": aborted, "video": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", default="base", choices=SUBSETS)
    ap.add_argument("--episode", type=int, default=19, help="episode number, 1-based")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--model", default="qwen3", choices=list(MODELS))
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--hold", type=float, default=0.8, help="seconds per step in the video")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--preview", default="", help="PNG rewritten after every step")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model = MODELS[args.model]
    err = check_server(model)
    if err:
        sys.exit(err)
    from embagent.planners import make_planner
    env, _, skills = make_env(args.subset)
    full = load_episodes(env, args.split, args.subset)
    planner = make_planner("s2m", skills, model_name=model, s2m={})
    out = args.out or os.path.join(HERE, "videos", f"live_{args.split}_{args.subset}_ep{args.episode}_{args.lang}.mp4")

    def on_step(s):
        print(f"step {s['step']:2d}: {s['action']:<40} {'ok' if s['ok'] else 'REJECTED'}  mode={s['mode']}  "
              f"distance={s['distance']:.2f} m", flush=True)

    r = run_live(env, full, args.episode - 1, planner, skills, model, args.lang, args.fps, args.hold, out,
                 on_frame=(lambda img: img.save(args.preview)) if args.preview else None, on_step=on_step)
    env.env.stop()
    print(f"{'SUCCESS' if r['success'] else 'FAILED'} after {r['steps']} steps, {r['distance']:.2f} m from the "
          f"target; {r['vlm_calls']} VLM calls")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
