#!/usr/bin/env bash
# Qwen3-VL-2B checkpoints for the model-size experiment: 4-bit AWQ (S2M; same quantizer and recipe as the 4B AWQ) and
# Qwen's official FP8 (EB prompt, as for the 4B). Plain HTTPS with range resume, so a stalled transfer continues.
set -uo pipefail
cd "$(dirname "$0")/.."
get() {  # repo file dest
  local url="https://huggingface.co/$1/resolve/main/$2" out="$3/$2"
  mkdir -p "$(dirname "$out")"
  for try in $(seq 1 40); do
    curl -sSL --fail -C - --speed-limit 20000 --speed-time 30 --retry 3 -o "$out.part" "$url" && { mv "$out.part" "$out"; return 0; }
    echo "$(date '+%T') retry $try $2"
    sleep 3
  done
  return 1
}
for spec in "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit models/Qwen3-VL-2B-Instruct-AWQ" \
            "Qwen/Qwen3-VL-2B-Instruct-FP8 models/Qwen3-VL-2B-Instruct-FP8"; do
  set -- $spec
  repo=$1
  dst=$2
  files=$(curl -s -m 30 "https://huggingface.co/api/models/$repo" | python3 -c \
    "import json,sys;print('\n'.join(s['rfilename'] for s in json.load(sys.stdin)['siblings'] if not s['rfilename'].startswith('.')))")
  for f in $files; do
    [ -f "$dst/$f" ] && continue
    get "$repo" "$f" "$dst" || { echo "FAILED $repo/$f"; exit 1; }
  done
  echo "$(date '+%F %T') $repo -> $dst done"
done
du -sh models/Qwen3-VL-2B-*
