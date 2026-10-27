"""Download a full model repo from ModelScope (much faster than HF from this network).

Usage: python modelscope_download.py <org/model> <out_dir> [revision]
Uses only the standard library so it runs before any env exists; skips files already complete.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev}&Recursive=true"
RESOLVE = "https://modelscope.cn/models/{repo}/resolve/{rev}/{path}"


def list_files(repo, rev):
    with urllib.request.urlopen(API.format(repo=repo, rev=rev), timeout=60) as r:
        data = json.load(r)
    if not data.get("Success", True) or "Data" not in data:
        raise RuntimeError(f"ModelScope listing failed for {repo}: {data.get('Message')}")
    return [f for f in data["Data"]["Files"] if f.get("Type") != "tree"]


def download(repo, out_dir, rev="master"):
    files = list_files(repo, rev)
    total = sum(f.get("Size", 0) for f in files)
    print(f"{repo}: {len(files)} files, {total / 1e9:.2f} GB -> {out_dir}", flush=True)
    for f in files:
        path, size = f["Path"], f.get("Size", 0)
        dst = os.path.join(out_dir, path)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = RESOLVE.format(repo=repo, rev=rev, path=urllib.parse.quote(path))
        tmp = dst + ".part"
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as w:
            while chunk := r.read(1 << 20):
                w.write(chunk)
        if size and os.path.getsize(tmp) != size:
            raise RuntimeError(f"size mismatch for {path}: {os.path.getsize(tmp)} != {size}")
        os.replace(tmp, dst)
        print(f"  ok {path} ({size / 1e6:.1f} MB)", flush=True)
    print(f"done {repo}", flush=True)


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "master")
