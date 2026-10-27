"""Measure how well a local VLM grounds EB-Navigation targets (dev split only).

For every dev episode and subset, renders the start view plus the 3 other 90-degree headings, and records:
presence probability, predicted box vs the simulator's instance box (IoU), and the depth-lifted 3D error of
the predicted box vs the ground-truth target position. Ground truth is used for scoring only.

  python scripts/eval_grounding.py --model Qwen2.5-VL-3B-Instruct-AWQ --subsets base common_sense visual_appearance
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ai2thor.controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

from embagent.grounding import Grounder, lift_box  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsets", nargs="*", default=["base", "common_sense", "visual_appearance"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"], help="test = official EB-Nav test episodes")
    ap.add_argument("--data", default="")
    ap.add_argument("--phrase", default="auto", choices=["auto", "instruction", "type"],
                    help="auto: target type for base, VLM-extracted phrase otherwise")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "grounding"))
    args = ap.parse_args()
    args.data = args.data or (os.path.join(ROOT, "third_party/EmbodiedBench/embodiedbench/envs/eb_navigation/datasets") if args.split == "test" else os.path.join(ROOT, "data", "eb_nav_dev"))

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.split == "dev" else "_test"
    out_path = os.path.join(args.out, f"{args.model}_{args.phrase}{tag}.jsonl")
    g = Grounder(args.model)
    ctrl = ai2thor.controller.Controller(agentMode="default", gridSize=0.1, visibilityDistance=10,
                                         renderDepthImage=True, renderInstanceSegmentation=True, width=500,
                                         height=500, fieldOfView=100, platform=CloudRendering, scene="FloorPlan1")
    t0 = time.time()
    with open(out_path, "w") as fout:
        for subset in args.subsets:
            tasks = json.load(open(os.path.join(args.data, f"{subset}.json")))["tasks"]
            for k, t in enumerate(tasks):
                ctrl.reset(scene=t["scene"])
                if args.phrase == "type" or (args.phrase == "auto" and subset == "base"):
                    phrase = t["targetObjectType"]
                elif args.phrase == "instruction":
                    phrase = t["instruction"]
                else:
                    phrase = g.target_phrase(t["instruction"])
                p = t["agentPose"]
                for r in range(4):
                    ev = ctrl.step(action="Teleport", position=p["position"],
                                   rotation={"x": 0, "y": (p["rotation"] + 90 * r) % 360, "z": 0},
                                   horizon=p["horizon"], standing=True)
                    a = ev.metadata["agent"]
                    pose = {"x": a["position"]["x"], "z": a["position"]["z"], "yaw": a["rotation"]["y"],
                            "horizon": a["cameraHorizon"]}
                    gt = ev.instance_detections2D.get(t["targetObjectIds"]) if ev.instance_detections2D else None
                    gt = [float(v) for v in gt] if gt is not None else None
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1]) if gt else 0.0
                    pv = g.presence(ev.frame, phrase)
                    box, raw = g.ground(ev.frame, phrase)
                    rec = {"subset": subset, "ep": k, "scene": t["scene"], "rot": r, "type": t["targetObjectType"],
                           "phrase": phrase, "gt_box": gt, "gt_area": gt_area, "p_vis": round(pv, 4),
                           "box": box, "raw": raw[:200]}
                    if box is not None:
                        rec["iou"] = round(iou(box, gt), 3) if gt else 0.0
                        lifted = lift_box(box, ev.depth_frame, pose)
                        if lifted:
                            tp = t["target_position"]
                            rec["lift_err"] = round(math.hypot(lifted[0] - tp["x"], lifted[1] - tp["z"]), 3)
                            rec["lift_range"] = round(lifted[3], 2)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                print(f"{subset} {k + 1}/{len(tasks)} phrase={phrase!r} ({time.time() - t0:.0f}s, {g.calls} calls)",
                      flush=True)
    ctrl.stop()
    summarize(out_path)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print(f"\n{path}: {len(recs)} frames")
    for subset in sorted({r["subset"] for r in recs}) + ["ALL"]:
        rs = [r for r in recs if subset in ("ALL", r["subset"])]
        vis = [r for r in rs if r["gt_area"] >= 100]
        inv = [r for r in rs if r["gt_area"] < 100]
        tp = [r for r in vis if r["box"] is not None and r.get("iou", 0) >= 0.3]
        fp_inv = [r for r in inv if r["box"] is not None]
        errs = sorted(r["lift_err"] for r in tp if "lift_err" in r)
        pv_vis = np.mean([r["p_vis"] for r in vis]) if vis else float("nan")
        pv_inv = np.mean([r["p_vis"] for r in inv]) if inv else float("nan")
        med = errs[len(errs) // 2] if errs else float("nan")
        p90 = errs[int(0.9 * (len(errs) - 1))] if errs else float("nan")
        print(f"{subset:18} visible={len(vis):3} hit@IoU.3={len(tp) / max(1, len(vis)):.2f} "
              f"box_on_invisible={len(fp_inv) / max(1, len(inv)):.2f} p_vis(vis)={pv_vis:.2f} p_vis(invis)={pv_inv:.2f} "
              f"lift_err med={med:.2f} p90={p90:.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
    else:
        main()
