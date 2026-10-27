"""RoboTHOR smoke probe on AI2-THOR 5.0: official LoCoBot config, teleport, actions, camera height, Stop check."""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from embagent.robothor.env import RoboTHOREnv  # noqa: E402

ep = json.load(gzip.open(os.path.join(ROOT, "data", "robothor", "dev", "FloorPlan_Train1_1.json.gz")))[0]
env = RoboTHOREnv()
obs = env.reset(ep)
ev = env.c.last_event
a = ev.metadata["agent"]
print("scene", ep["scene"], "target", ep["object_type"], "frame", obs["rgb"].shape, "depth", obs["depth"].shape)
print("agent y", round(a["position"]["y"], 3), "camera y", round(ev.metadata["cameraPosition"]["y"], 3),
      "floor y (path)", round(ep["shortest_path"][0]["y"], 3), "-> cam_h", round(env.cam_h, 3),
      "horizon", a["cameraHorizon"])
for act in ["MoveAhead", "RotateRight", "MoveAhead", "LookUp", "LookDown", "RotateLeft"]:
    obs, done, info = env.step(act)
    p = obs["pose"]
    print(f"{act:12} ok={info['ok']} pos=({p['x']:.3f},{p['z']:.3f}) yaw={p['yaw']:.2f} hor={p['horizon']:.1f}")
print("target-type objects:", [(round(o["position"]["x"], 2), round(o["position"]["z"], 2))
                               for o in ev.metadata["objects"] if o["objectType"] == ep["object_type"]])
print("dist to nearest target", env.target_distance(), "visible now", env.target_visible())
env.close()
print("OK")
