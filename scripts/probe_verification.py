"""Absolute vs. relative verification for small VLMs (dev split only).

For each dev frame (4 headings per episode) with the target visible: ask the VLM for every candidate box,
then score each candidate two ways:
  absolute  - presence probability on a magnified crop of that box ("does this show <target>?")
  relative  - Set-of-Mark choice probability among all candidates in the frame
and measure how well each separates the true target box (IoU >= 0.3 with the simulator box) from the others
(ROC-AUC over candidates, top-1 accuracy per frame). Ground truth is used only for scoring.

  python scripts/probe_verification.py --model Qwen3-VL-4B-Instruct-AWQ --subsets visual_appearance common_sense
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ai2thor.controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

from embagent.grounding import Grounder, _iou  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsets", nargs="*", default=["visual_appearance", "common_sense", "complex_instruction"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "grounding"))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    data_dir = os.path.join(ROOT, "third_party/EmbodiedBench/embodiedbench/envs/eb_navigation/datasets") if args.split == "test" else os.path.join(ROOT, "data", "eb_nav_dev")
    tag = "" if args.split == "dev" else "_test"
    g = Grounder(args.model)
    c = ai2thor.controller.Controller(agentMode="default", gridSize=0.1, visibilityDistance=10, renderDepthImage=False,
                                      renderInstanceSegmentation=True, width=500, height=500, fieldOfView=100,
                                      platform=CloudRendering, scene="FloorPlan1")
    out = open(os.path.join(args.out, f"verify_{args.model}{tag}.jsonl"), "w")
    for sub in args.subsets:
        for t in json.load(open(os.path.join(data_dir, f"{sub}.json")))["tasks"]:
            c.reset(scene=t["scene"])
            q = g.target_spec(t["instruction"])["query"]
            p = t["agentPose"]
            for r in range(4):
                ev = c.step(action="Teleport", position=p["position"], rotation={"x": 0, "y": (p["rotation"] + 90 * r) % 360, "z": 0},
                            horizon=p["horizon"], standing=True)
                gt = (ev.instance_detections2D or {}).get(t["targetObjectIds"])
                if gt is None or (gt[2] - gt[0]) * (gt[3] - gt[1]) < 100:
                    continue
                gt = [float(v) for v in gt]
                single, _ = g.ground(ev.frame, q)
                boxes = g.ground_all(ev.frame, q)
                if single is not None and all(_iou(single, b) < 0.5 for b in boxes):
                    boxes = (boxes + [single])[:6]
                rec = {"subset": sub, "scene": t["scene"], "rot": r, "query": q,
                       "single_hit": bool(single is not None and _iou(single, gt) >= 0.3), "n_cand": len(boxes)}
                if boxes:
                    rec["is_true"] = [bool(_iou(b, gt) >= 0.3) for b in boxes]
                    rec["abs"] = [round(g.zoom_presence(ev.frame, b, q), 4) for b in boxes]
                    rec["rel"] = [round(float(x), 4) for x in g.choose(ev.frame, boxes, q)[1:]]
                out.write(json.dumps(rec) + "\n")
                out.flush()
            print(sub, t["scene"], "done", flush=True)
    c.stop()
    out.close()
    report(os.path.join(args.out, f"verify_{args.model}{tag}.jsonl"))


def report(path):
    recs = [json.loads(line) for line in open(path)]
    for sub in sorted({r["subset"] for r in recs}) + ["ALL"]:
        rs = [r for r in recs if sub in ("ALL", r["subset"])]
        with_true = [r for r in rs if any(r.get("is_true", []))]
        multi = [r for r in with_true if r["n_cand"] >= 2]
        pos_a, neg_a, pos_r, neg_r = [], [], [], []
        for r in multi:
            for t, a, rel in zip(r["is_true"], r["abs"], r["rel"]):
                (pos_a if t else neg_a).append(a)
                (pos_r if t else neg_r).append(rel)
        top_abs = np.mean([r["is_true"][int(np.argmax(r["abs"]))] for r in multi]) if multi else float("nan")
        top_rel = np.mean([r["is_true"][int(np.argmax(r["rel"]))] for r in multi]) if multi else float("nan")
        print(f"{sub:20} visible frames={len(rs):3} single-box hit={np.mean([r['single_hit'] for r in rs]):.2f} "
              f"target in candidates={len(with_true) / max(1, len(rs)):.2f} | multi-candidate frames={len(multi):3} "
              f"top-1 abs={top_abs:.2f} rel={top_rel:.2f} | AUC abs={auc(pos_a, neg_a):.2f} rel={auc(pos_r, neg_r):.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main()
