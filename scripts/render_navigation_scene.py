#!/usr/bin/env python3
"""Render real iTHOR scene frames from one saved navigation trajectory.

This is a deterministic visual replay utility, not an evaluation runner: it
never calls a VLM or executes a policy. It teleports an iTHOR agent to each
recorded pose in the saved trajectory and writes RGB, metric-depth, and a
manifest that links every frame to its original result record.

Example (use the dedicated rendering environment created for this project):
  /home/ht/.conda/envs/s2m-scene-viz/bin/python scripts/render_navigation_scene.py \
    --run runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1 \
    --subset long_horizon --episode 42 --output outputs/s2m_long_42_scene
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_episode(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("trajectory"):
        raise ValueError(f"{path} is not a formal episode summary with a trajectory")
    return value


def teleport(controller: Any, pose: dict[str, Any], agent_y: float) -> Any:
    return controller.step(
        action="Teleport",
        position={"x": float(pose["x"]), "y": agent_y, "z": float(pose["z"])},
        rotation={"x": 0.0, "y": float(pose.get("yaw", 0.0)), "z": 0.0},
        horizon=float(pose.get("horizon", 0.0)),
        standing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RGB-D frames from a saved iTHOR trajectory.")
    parser.add_argument("--run", type=Path, required=True, help="formal result directory")
    parser.add_argument("--subset", required=True, help="subset directory, e.g. long_horizon")
    parser.add_argument("--episode", type=int, required=True, help="episode number within the subset")
    parser.add_argument("--output", type=Path, required=True, help="directory for rendered frames")
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--agent-y", type=float, default=0.900999, help="recorded iTHOR camera base height")
    args = parser.parse_args()

    source = args.run / args.subset / f"episode_{args.episode}.json"
    record = load_episode(source)
    trajectory = record["trajectory"]
    if any("x" not in p or "z" not in p for p in trajectory):
        raise ValueError("trajectory lacks metric x/z poses; use a formal result summary, not _eb_logs")

    try:
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
    except ImportError as exc:
        raise SystemExit(
            "ai2thor is required. Use /home/ht/.conda/envs/s2m-scene-viz/bin/python or install ai2thor==5.0.0."
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, depth_dir = args.output / "rgb", args.output / "depth_mm"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    controller = Controller(
        platform=CloudRendering,
        agentMode="default",
        scene=record["scene"],
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=True,
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
    )
    rendered = []
    try:
        for step, pose in enumerate(trajectory):
            event = teleport(controller, pose, args.agent_y)
            if not event.metadata.get("lastActionSuccess", False):
                raise RuntimeError(f"Teleport failed at step {step}: {event.metadata.get('errorMessage', '')}")
            rgb_name = f"step_{step:03d}.png"
            depth_name = f"step_{step:03d}.png"
            Image.fromarray(event.frame).save(rgb_dir / rgb_name)
            depth_mm = np.clip(event.depth_frame, 0.0, 65.535) * 1000.0
            Image.fromarray(depth_mm.astype(np.uint16)).save(depth_dir / depth_name)
            rendered.append({
                "step": step,
                "rgb": f"rgb/{rgb_name}",
                "depth_mm": f"depth_mm/{depth_name}",
                "recorded_pose": {key: pose.get(key) for key in ("x", "z", "yaw", "horizon")},
                "recorded_action": pose.get("action"),
                "recorded_distance": pose.get("dist"),
                "recorded_visible": pose.get("vis"),
            })
    finally:
        controller.stop()

    manifest = {
        "kind": "iTHOR scene reconstruction from saved trajectory",
        "source_record": str(source),
        "scene": record["scene"],
        "instruction": record.get("instruction"),
        "task_success": record.get("task_success"),
        "rendering": {"width": args.width, "height": args.height, "field_of_view": args.fov,
                      "agent_y": args.agent_y, "depth_unit": "uint16 millimetres"},
        "caveat": "Frames reconstruct the saved scene name and pose using the installed iTHOR build. They are not original saved camera frames; task-time object-hide state is unavailable in formal result summaries.",
        "frames": rendered,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {len(rendered)} RGB-D frames to {args.output}")
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
