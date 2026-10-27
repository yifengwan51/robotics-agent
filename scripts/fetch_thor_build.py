"""Fetch an AI2-THOR Unity build in many small parallel range requests, verify sha256, and install it where
ai2thor looks (~/.ai2thor/releases/<name>) so the Controller skips its own (single-connection) download.

Reuses partNN files from scripts/fetch_thor_build.sh if present. Standard library only.
  python3 scripts/fetch_thor_build.py [CloudRendering|Linux64] [commit]
"""
import concurrent.futures as cf
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

PLATFORM = sys.argv[1] if len(sys.argv) > 1 else "CloudRendering"
COMMIT = sys.argv[2] if len(sys.argv) > 2 else "f0825767cd50d69f666c7f282e54abfe58f1e917"
NAME = f"thor-{PLATFORM}-{COMMIT}"
URL = f"http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/{NAME}.zip"
REL = os.path.expanduser("~/.ai2thor/releases")
TMP = os.path.expanduser("~/.ai2thor/tmp")
CHUNK, THREADS, OLD_PARTS = 4 << 20, 16, 8


def get(url, rng=None, timeout=60):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(path, start, end):
    for attempt in range(8):
        try:
            data = get(URL, (start, end))
            if len(data) == end - start + 1:
                with open(path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
        except Exception as e:
            print(f"retry {start}-{end} ({attempt + 1}): {e}", flush=True)
    raise RuntimeError(f"failed range {start}-{end}")


def main():
    if os.path.isdir(os.path.join(REL, NAME)):
        print(NAME, "already installed")
        return
    os.makedirs(REL, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    sha = get(URL[:-4] + ".sha256").decode().strip()
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    work = tempfile.mkdtemp(dir=TMP, prefix="fetchpy.")
    out = os.path.join(work, "build.zip")
    with open(out, "wb") as f:
        f.truncate(size)

    # Reuse finished or partial pieces from the shell fetcher (equal ranges of ceil(size/8) bytes).
    part = -(-size // OLD_PARTS)
    have = []  # (start, end) byte ranges already present
    for d in glob.glob(os.path.join(TMP, "fetch.*")):
        for i in range(OLD_PARTS):
            p = os.path.join(d, f"part{i}")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                start = i * part
                n = min(os.path.getsize(p), min(size, start + part) - start)
                with open(p, "rb") as src, open(out, "r+b") as dst:
                    dst.seek(start)
                    dst.write(src.read(n))
                have.append((start, start + n - 1))
    covered = sum(e - s + 1 for s, e in have)
    print(f"{NAME}: {size / 2**20:.0f} MB, reused {covered / 2**20:.0f} MB", flush=True)

    todo, pos = [], 0
    for s, e in sorted(have) + [(size, size)]:
        while pos < s:
            todo.append((pos, min(s, pos + CHUNK) - 1))
            pos = min(s, pos + CHUNK)
        pos = max(pos, e + 1)
    done = 0
    with cf.ThreadPoolExecutor(THREADS) as ex:
        for n in ex.map(lambda r: fetch(out, *r), todo):
            done += n
            if done % (40 << 20) < CHUNK:
                print(f"  {(covered + done) / 2**20:.0f}/{size / 2**20:.0f} MB", flush=True)

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    if h.hexdigest() != sha:
        raise SystemExit(f"sha256 mismatch: {h.hexdigest()} != {sha}")
    xdir = os.path.join(work, "x")
    zipfile.ZipFile(out).extractall(xdir)
    os.rename(xdir, os.path.join(REL, NAME))
    for exe in glob.glob(os.path.join(REL, NAME, "thor-*")):
        os.chmod(exe, 0o755)
    shutil.rmtree(work, ignore_errors=True)
    for d in glob.glob(os.path.join(TMP, "fetch.*")):
        shutil.rmtree(d, ignore_errors=True)
    print("installed", os.path.join(REL, NAME), flush=True)


if __name__ == "__main__":
    main()
