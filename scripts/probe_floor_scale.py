"""Probe: does a per-frame scale fitted to the floor fix monocular depth better than one global scale?
Designed on DEV; FLOOR_PROBE_DATA=<dataset dir> re-measures the frozen method on other start views (e.g. TEST).

Renders dev start views (4 headings), queries a running depth server for the raw prediction, and compares
  global   : the DEV calibration file's single scale
  floor    : per-frame scale that puts the floor at the known camera height (no simulator information)
  oracle   : per-frame median scale against simulator depth (upper bound for any per-frame scale)
on AbsRel and on the occupancy labels the planner uses (floor if lifted height < 0.10 m, obstacle up to 1.9 m).

  DEPTH_PORT=8011 python scripts/probe_floor_scale.py DA2-Metric-Indoor-Base data/depth_calib_da2_base.json
(env embnav_thor). Raw frames are cached in /tmp/floor_probe_<model>.npz, so re-analysis needs no server.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from embagent.geometry import CAM_HEIGHT, focal  # noqa: E402

FOV, SIZE = 100.0, 500
DATA = os.environ.get("FLOOR_PROBE_DATA") or os.path.join(ROOT, "data", "eb_nav_dev")
TAG = "" if DATA.rstrip("/").endswith("eb_nav_dev") else "_" + os.path.basename(os.path.dirname(DATA.rstrip("/")))


def render(model):
    import ai2thor.controller
    from ai2thor.platform import CloudRendering
    from embagent.depth_est import estimate_raw
    url = f"http://127.0.0.1:{os.environ.get('DEPTH_PORT', '8010')}/"
    tasks = json.load(open(os.path.join(DATA, "base.json")))["tasks"]
    c = ai2thor.controller.Controller(agentMode="default", gridSize=0.1, visibilityDistance=10, renderDepthImage=True,
                                      width=SIZE, height=SIZE, fieldOfView=FOV, platform=CloudRendering,
                                      scene="FloorPlan1")
    P, G = [], []
    for t in tasks:
        c.reset(scene=t["scene"])
        p = t["agentPose"]
        for r in (0, 90, 180, 270):
            ev = c.step(action="Teleport", position=p["position"], horizon=p["horizon"], standing=True,
                        rotation={"x": 0, "y": (p["rotation"] + r) % 360, "z": 0})
            P.append(estimate_raw(ev.frame, url=url).astype(np.float16))
            G.append(ev.depth_frame.astype(np.float16))
    c.stop()
    return np.stack(P), np.stack(G)


V_DOWN = (np.mgrid[0:SIZE, 0:SIZE][0] - SIZE / 2) / focal(SIZE, FOV)  # image-down ray slope per unit planar depth


def height(d):
    """Height above the floor of every pixel for planar depth d at horizon 0 (the dev start views)."""
    return CAM_HEIGHT - V_DOWN * d


def floor_scale(raw, lo=0.55):
    """Per-frame scale from pixels in the lower image whose rays point down: the floor gives s = h_cam / (v * d),
    anything standing on it gives a larger value, so the scale is the mode of the lower cluster."""
    m = (V_DOWN > np.tan(np.radians(15))) & (raw > 0.05)
    m[: int(lo * SIZE)] = False
    if m.sum() < 500:
        return None
    s = CAM_HEIGHT / (V_DOWN[m] * raw[m])
    ls = np.log(s)
    hist, edges = np.histogram(ls, bins=80, range=(np.log(0.1), np.log(2.0)))
    k = int(np.argmax(hist))
    if hist[k] < 0.05 * m.sum():
        return None
    sel = np.abs(ls - (edges[k] + edges[k + 1]) / 2) < 0.08
    return float(np.exp(np.median(ls[sel])))


def labels(d):
    h = height(d)
    ok = (d > 0.1) & (d < 5.0)
    return np.where(~ok, 0, np.where(h < 0.10, 1, np.where(h <= 1.9, 2, 0)))  # 0 ignore, 1 floor, 2 obstacle


def main():
    model, calib = sys.argv[1], sys.argv[2]
    cache = f"/tmp/floor_probe_{model}{TAG}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        P, G = z["P"], z["G"]
    else:
        P, G = render(model)
        np.savez(cache, P=P, G=G)
    P, G = P.astype(np.float32), G.astype(np.float32)
    g_scale = json.load(open(calib))["scale"]
    res = {k: {"absrel": [], "floor_as_obst": [], "obst_as_floor": [], "scale": []} for k in ("global", "floor", "oracle")}
    fallback = 0
    for raw, gt in zip(P, G):
        valid = (gt > 0.2) & (gt < 8.0)
        from embagent.depth_est import floor_scale as frozen
        fs = frozen(raw, g_scale)
        fs = None if fs == g_scale else fs
        if fs is None:
            fallback += 1
            fs = g_scale
        scales = {"global": g_scale, "floor": fs, "oracle": float(np.median(gt[valid] / raw[valid]))}
        lg = labels(gt)
        for k, s in scales.items():
            d = raw * s
            res[k]["absrel"].append(float(np.mean(np.abs(d[valid] - gt[valid]) / gt[valid])))
            lp = labels(d)
            res[k]["floor_as_obst"].append(float(np.mean(lp[lg == 1] == 2)) if (lg == 1).any() else np.nan)
            res[k]["obst_as_floor"].append(float(np.mean(lp[lg == 2] == 1)) if (lg == 2).any() else np.nan)
            res[k]["scale"].append(s)
    print(f"{model}: {len(P)} frames from {DATA}, floor fit fell back to the global scale on {fallback}")
    print(f"{'scale':8s} {'AbsRel':>7s} {'floor->obst':>12s} {'obst->floor':>12s} {'scale CV':>9s}")
    for k, r in res.items():
        print(f"{k:8s} {np.mean(r['absrel']):7.3f} {100 * np.nanmean(r['floor_as_obst']):11.1f}% "
              f"{100 * np.nanmean(r['obst_as_floor']):11.1f}% {np.std(r['scale']) / np.mean(r['scale']):9.3f}")
    summary = {k: {"absrel": float(np.mean(r["absrel"])), "floor_as_obstacle": float(np.nanmean(r["floor_as_obst"])),
                   "obstacle_as_floor": float(np.nanmean(r["obst_as_floor"]))} for k, r in res.items()}
    summary.update(frames=len(P), fallback=fallback, model=model, data=os.path.relpath(DATA, ROOT))
    out = os.path.join(ROOT, "paper", "tables", f"floor_probe{TAG or '_dev'}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print("wrote", out)
    o, f = np.array(res["oracle"]["scale"]), np.array(res["floor"]["scale"])
    print(f"floor vs oracle scale: median |log ratio| {np.median(np.abs(np.log(f / o))):.3f}, "
          f"corr {np.corrcoef(f, o)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
