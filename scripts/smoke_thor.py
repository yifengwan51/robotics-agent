"""AI2-THOR smoke test: downloads the CloudRendering build on first run, renders one EB-Navigation scene,
steps a few actions, and reports timing. Usage: python scripts/smoke_thor.py [--platform cloud|linux64]"""
import argparse
import os
import time

import ai2thor.controller
from ai2thor.platform import CloudRendering, Linux64
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--platform", default="cloud", choices=["cloud", "linux64"])
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
args = ap.parse_args()

t0 = time.time()
ctrl = ai2thor.controller.Controller(
    agentMode="default", gridSize=0.1, visibilityDistance=10, renderDepthImage=True,
    renderInstanceSegmentation=True, width=500, height=500, fieldOfView=100,
    platform=CloudRendering if args.platform == "cloud" else Linux64, scene="FloorPlan11")
print(f"controller up in {time.time() - t0:.1f}s (includes first-time build download)", flush=True)

t0 = time.time()
for action in ["MoveAhead", "RotateRight", "MoveAhead", "LookDown", "MoveLeft"]:
    ev = ctrl.step(action=action)
    print(action, "ok" if ev.metadata["lastActionSuccess"] else f"failed: {ev.metadata['errorMessage']}")
print(f"5 steps in {time.time() - t0:.2f}s; frame {ev.frame.shape}, depth {ev.depth_frame.shape}", flush=True)
os.makedirs(args.out, exist_ok=True)
Image.fromarray(ev.frame).save(os.path.join(args.out, f"smoke_thor_{args.platform}.png"))
ctrl.stop()
print("OK")
