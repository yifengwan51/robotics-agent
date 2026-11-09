"""Let the agent see images in this environment.

The API gateway configured in ~/.claude-robot-opus55/settings.json drops images inside tool results (the Read tool
returns them that way), but forwards images in ordinary user messages. This script sends the given images as
ordinary message content to the same model through the same gateway and prints what the model sees.

  ~/miniforge3/envs/embnav/bin/python tools/look.py IMAGE [IMAGE ...] -q "question"
  IMAGE: .png/.jpg, or .pdf (page 1 is rendered; use file.pdf:N for page N)
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

from PIL import Image

CFG = os.path.expanduser("~/.claude-robot-opus55/settings.json")
MAX_SIDE = 1568  # long edge the API uses without further downscaling


def load(path):
    page = 1
    if ".pdf:" in path:
        path, page = path.rsplit(":", 1)
        page = int(page)
    if path.lower().endswith(".pdf"):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", "-f", str(page), "-l", str(page), path,
                            os.path.join(d, "p")], check=True)
            im = Image.open(os.path.join(d, "p.png")).convert("RGB")
    else:
        im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()
    env = json.load(open(CFG))["env"]
    url, tok = env["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages", env["ANTHROPIC_AUTH_TOKEN"]
    content = []
    for p in a.images:
        data, size = load(p)
        content.append({"type": "text", "text": f"Image: {os.path.basename(p)} ({size[0]}x{size[1]} px)"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    content.append({"type": "text", "text": a.question})
    body = {"model": env.get("ANTHROPIC_MODEL", "claude-opus-5-5"), "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": tok,
        "Authorization": "Bearer " + tok})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:400]!r}")
    print("".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text"))


if __name__ == "__main__":
    main()
