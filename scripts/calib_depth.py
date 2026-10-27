"""Fit the global calibration of the monocular depth model on DEV frames only (never test).

Decides whether the network predicts planar or ray depth for this camera (100 deg FOV) and a single scale
factor, by comparing against simulator depth on the dev start views. Writes data/depth_calib.json and prints
error statistics (AbsRel, delta<1.25) before and after calibration.

  python scripts/calib_depth.py        (env embnav_thor; depth server running)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ai2thor.controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

from embagent.depth_est import CALIB, _ray_to_planar, estimate_raw  # noqa: E402


def stats(pred, gt):
    m = (gt > 0.2) & (gt < 8.0)
    p, g = pred[m], gt[m]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    return absrel, float(np.mean(ratio < 1.25))


def main():
    tasks = json.load(open(os.path.join(ROOT, "data", "eb_nav_dev", "base.json")))["tasks"]
    c = ai2thor.controller.Controller(agentMode="default", gridSize=0.1, visibilityDistance=10, renderDepthImage=True,
                                      width=500, height=500, fieldOfView=100, platform=CloudRendering, scene="FloorPlan1")
    preds, gts = [], []
    for t in tasks:
        c.reset(scene=t["scene"])
        p = t["agentPose"]
        for r in (0, 180):
            ev = c.step(action="Teleport", position=p["position"], rotation={"x": 0, "y": (p["rotation"] + r) % 360, "z": 0},
                        horizon=p["horizon"], standing=True)
            preds.append(estimate_raw(ev.frame))
            gts.append(ev.depth_frame.astype(np.float32))
    c.stop()
    P, G = np.stack(preds), np.stack(gts)
    conv = _ray_to_planar(500, 100.0)[None]
    best = None
    for ray in (False, True):
        Q = P * conv if ray else P
        m = (G > 0.2) & (G < 8.0)
        scale = float(np.median(G[m] / Q[m]))
        a, d = stats(Q * scale, G)
        print(f"ray={ray}: scale={scale:.3f}  AbsRel={a:.3f}  delta<1.25={d:.3f}")
        if best is None or a < best[0]:
            best = (a, {"scale": scale, "ray": ray, "absrel_dev": a, "delta125_dev": d, "frames": len(P)})
    a0, d0 = stats(P, G)
    print(f"uncalibrated: AbsRel={a0:.3f} delta<1.25={d0:.3f}")
    os.makedirs(os.path.dirname(CALIB), exist_ok=True)
    json.dump(best[1], open(CALIB, "w"), indent=1)
    print("wrote", CALIB, best[1])


if __name__ == "__main__":
    main()
