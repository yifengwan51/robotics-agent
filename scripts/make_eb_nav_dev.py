"""Build a DEV split for EB-Navigation from the 30 iTHOR rooms the official test set does not use.

EmbodiedBench ships target objects (Gen_data/Floorplan2Object.json) and the three instruction variants
(instruct-commonsense/-complex/-visual.json) for 90 rooms; the official test subsets use FloorPlan11-30,
211-230, 311-330. This script reproduces Gen_data/base_task_gen.py (start pose 2.5-3.0 m from the first
target instance, horizon 0, from GetInteractablePoses) on FloorPlan1-10, 201-210, 301-310, then derives the
five subsets exactly as the authors did (long_horizon = base with the start yaw rotated by 180 degrees).

Usage (env embnav_thor): python scripts/make_eb_nav_dev.py [--seed 0]
Writes data/eb_nav_dev/{base,common_sense,complex_instruction,visual_appearance,long_horizon}.json
"""
import argparse
import json
import math
import os
import random

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "third_party/EmbodiedBench/embodiedbench/envs/eb_navigation/Gen_data")
TEST_DATA = os.path.join(ROOT, "third_party/EmbodiedBench/embodiedbench/envs/eb_navigation/datasets")
DEV_SCENES = [f"FloorPlan{i}" for i in [*range(1, 11), *range(201, 211), *range(301, 311)]]
MIN_D, MAX_D = 2.5, 3.0


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eb_nav_dev"))
    args = ap.parse_args()
    random.seed(args.seed)

    test_scenes = {t["scene"] for t in json.load(open(os.path.join(TEST_DATA, "base.json")))["tasks"]}
    assert not test_scenes & set(DEV_SCENES), "dev scenes overlap the test set"
    mapping = json.load(open(os.path.join(GEN, "Floorplan2Object.json")))
    instr = {k: json.load(open(os.path.join(GEN, f))) for k, f in [
        ("common_sense", "instruct-commonsense.json"), ("complex_instruction", "instruct-complex.json"),
        ("visual_appearance", "instruct-visual.json")]}

    ctrl = Controller(agentMode="default", visibilityDistance=5, scene="FloorPlan1", gridSize=0.1,
                      renderDepthImage=False, renderInstanceSegmentation=False, width=300, height=300,
                      fieldOfView=90, platform=CloudRendering)
    base = []
    for scene in DEV_SCENES:
        target_type = mapping[scene]
        ctrl.reset(scene=scene)
        objs = [o for o in ctrl.last_event.metadata["objects"] if o["objectType"] == target_type]
        if not objs:
            print("no", target_type, "in", scene)
            continue
        target = objs[0]
        poses = ctrl.step(action="GetInteractablePoses", objectId=target["objectId"], horizons=[0]).metadata["actionReturn"] or []
        random.shuffle(poses)
        pose = next((p for p in poses if MIN_D <= dist(p, target["position"]) <= MAX_D), None)
        if pose is None:
            print("no valid start pose in", scene)
            continue
        base.append({
            "targetObjectType": target_type, "targetObjectIds": target["objectId"],
            "target_position": target["position"],
            "agentPose": {"position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
                          "rotation": pose["rotation"], "horizon": pose["horizon"]},
            "scene": scene, "object_to_hide": [o["objectId"] for o in objs[1:]],
            "instruction": f"navigate to the {target_type} in the room and be as close as possible to it"})
        print(scene, target_type, f"start {dist(pose, target['position']):.2f} m")
    ctrl.stop()

    os.makedirs(args.out, exist_ok=True)
    subsets = {"base": base}
    for name, table in instr.items():
        subsets[name] = [dict(t, instruction=table[t["scene"]]) for t in base]
    subsets["long_horizon"] = [dict(t, agentPose=dict(t["agentPose"], rotation=float((t["agentPose"]["rotation"] + 180) % 360)))
                               for t in base]
    for name, tasks in subsets.items():
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump({"tasks": tasks}, f, indent=1)
    print(f"wrote {len(base)} episodes x {len(subsets)} subsets -> {args.out}")


if __name__ == "__main__":
    main()
