"""Monocular metric depth (Depth Anything V2 Metric-Indoor-Small) as a tiny local HTTP service.

Used for the "no depth sensor" robustness study: the agent gets estimated instead of ground-truth depth.
The server runs in the `vlm` env (torch + transformers), on CPU by default so it does not compete with vLLM
for the 8 GB GPU; the client is stdlib + numpy only, so the simulator env needs no extra packages.

  ~/miniforge3/envs/vlm/bin/python -m embagent.depth_est --serve [--device cpu] [--threads 8]
"""
import argparse
import io
import json
import os
import threading
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("DEPTH_PORT", "8010"))
# DEPTH_MODEL selects the checkpoint under models/ (server side); DEPTH_CALIB the DEV calibration file (client side)
MODEL = os.path.join(ROOT, "models", os.environ.get("DEPTH_MODEL", "DA2-Metric-Indoor-Small"))
CALIB = os.environ.get("DEPTH_CALIB") or os.path.join(ROOT, "data", "depth_calib.json")


def serve(device="cpu", threads=8):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    torch.set_num_threads(threads)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # identity: which checkpoint this server runs
            body = os.path.basename(MODEL).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            img = Image.open(io.BytesIO(self.rfile.read(n))).convert("RGB")
            with lock, torch.no_grad():
                inp = proc(images=img, return_tensors="pt").to(device)
                pred = model(**inp).predicted_depth[:, None]
                pred = torch.nn.functional.interpolate(pred, size=img.size[::-1], mode="bicubic",
                                                       align_corners=False)[0, 0]
            buf = io.BytesIO()
            np.save(buf, pred.float().cpu().numpy().astype(np.float32))
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"depth server on 127.0.0.1:{PORT} ({device}, {threads} threads)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


def estimate_raw(rgb, url=None):
    """Raw network output (metres, model convention) for an HxWx3 uint8 frame."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    req = urllib.request.Request(url or f"http://127.0.0.1:{PORT}/", data=buf.getvalue(), method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.load(io.BytesIO(r.read()))


def served_model(url=None):
    """Checkpoint name reported by the running server; servers started before the GET endpoint existed ran Small."""
    try:
        with urllib.request.urlopen(url or f"http://127.0.0.1:{PORT}/", timeout=10) as r:
            return r.read().decode()
    except urllib.error.HTTPError:
        return "DA2-Metric-Indoor-Small"


_calib = None


def _ray_to_planar(size, fov):
    c = (size - 1) / 2
    f = (size / 2) / np.tan(np.radians(fov) / 2)
    v, u = np.mgrid[0:size, 0:size]
    return f / np.sqrt(f ** 2 + (u - c) ** 2 + (v - c) ** 2)


def estimate(rgb, fov=100.0):
    """Planar metric depth with the global calibration fitted once on DEV frames (scripts/calib_depth.py)."""
    global _calib
    if _calib is None:
        _calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {"scale": 1.0, "ray": False}
    d = estimate_raw(rgb) * _calib["scale"]
    if _calib.get("ray"):
        d = d * _ray_to_planar(rgb.shape[0], fov)
    return d.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.serve:
        serve(a.device, a.threads)
