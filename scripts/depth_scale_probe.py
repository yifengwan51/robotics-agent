"""DEV-only probe: how much of the monocular-depth error is a per-frame scale error, and can a floor prior fix it?

Renders the 60 DEV start views used by scripts/calib_depth.py and compares against simulator depth:
  global  - the single scale fitted on DEV (data/depth_calib.json), as used by the depth_est_r1 TEST run
  oracle  - per-frame median scale from simulator depth: upper bound for any per-frame scale correction
  floor   - per-frame scale from the floor plane, no simulator depth used: with known camera height and tilt, a
            floor pixel in row v has planar depth h*f/(v-cy); scale = a low percentile of expected/predicted over
            the bottom image band (obstacles are nearer than the floor, so they only raise the ratio)
  python scripts/depth_scale_probe.py      (env embnav_thor; depth server running: python -m embagent.depth_est --serve)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ai2thor.controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

from embagent.depth_est import CALIB, estimate_raw  # noqa: E402
from embagent.geometry import CAM_HEIGHT  # noqa: E402

SIZE, FOV = 500, 100.0


def stats(pred, gt):
    m = (gt > 0.2) & (gt < 8.0)
    p, g = pred[m], gt[m]
    return float(np.mean(np.abs(p - g) / g)), float(np.mean(np.maximum(p / g, g / p) < 1.25))


def floor_depth(band=0.8):
    """Planar depth of the floor for rows in the bottom band (horizon 0), NaN elsewhere."""
    f = (SIZE / 2) / np.tan(np.radians(FOV) / 2)
    cy = (SIZE - 1) / 2
    v = np.arange(SIZE, dtype=float)
    z = np.where(v >= band * SIZE, CAM_HEIGHT * f / np.maximum(v - cy, 1e-6), np.nan)
    return np.repeat(z[:, None], SIZE, axis=1)


def main():
    tasks = json.load(open(os.path.join(ROOT, "data", "eb_nav_dev", "base.json")))["tasks"]
    c = ai2thor.controller.Controller(agentMode="default", gridSize=0.1, visibilityDistance=10, renderDepthImage=True,
                                      width=SIZE, height=SIZE, fieldOfView=FOV, platform=CloudRendering,
                                      scene="FloorPlan1")
    preds, gts = [], []
    for t in tasks:
        c.reset(scene=t["scene"])
        p = t["agentPose"]
        for r in (0, 180):
            ev = c.step(action="Teleport", position=p["position"], horizon=p["horizon"], standing=True,
                        rotation={"x": 0, "y": (p["rotation"] + r) % 360, "z": 0})
            preds.append(estimate_raw(ev.frame))
            gts.append(ev.depth_frame.astype(np.float32))
    c.stop()
    P, G = np.stack(preds), np.stack(gts)
    g_scale = json.load(open(CALIB))["scale"]
    Z = floor_depth()
    zm = ~np.isnan(Z)
    rows = {"global": [], "oracle": [], "floor_p20": [], "floor_p50": []}
    sanity = []
    for p, g in zip(P, G):
        m = (g > 0.2) & (g < 8.0)
        rows["global"].append(p * g_scale)
        rows["oracle"].append(p * float(np.median(g[m] / p[m])))
        ratio = (Z / p)[zm]
        for q in (20, 50):
            rows[f"floor_p{q}"].append(p * float(np.percentile(ratio, q)))
        sanity.append(float(np.median((g / Z)[zm])))  # ~1 if the bottom band is floor and CAM_HEIGHT is right
    print(f"{len(P)} DEV frames; camera height {CAM_HEIGHT} m; median GT/expected-floor over bottom band "
          f"{np.median(sanity):.3f} (frames within 10%: {np.mean(np.abs(np.array(sanity) - 1) < 0.1):.2f})")
    for name, ds in rows.items():
        a, d = stats(np.stack(ds), G)
        print(f"  {name:10} AbsRel {a:.3f}  delta<1.25 {d:.3f}")


if __name__ == "__main__":
    main()
