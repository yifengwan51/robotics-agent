#!/usr/bin/env bash
# One entry point for the demo. Run from anywhere:
#   bash demo/run_demo.sh show                          live browser view: EB prompt vs S2M on the same local model
#   bash demo/run_demo.sh replay long_horizon 14        side-by-side video of one saved test episode (no model needed)
#   bash demo/run_demo.sh gallery                       the three episodes of the paper's trajectory figure
#   bash demo/run_demo.sh serve                         only start the local VLM server
#   bash demo/run_demo.sh stop                          stop the live view, the VLM server, and leftover simulators
# Extra options go to the Python scripts through DEMO_ARGS, e.g. DEMO_ARGS="--lang en".
set -uo pipefail
DEMO="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DEMO")"
PY=~/miniforge3/envs/embnav_thor/bin/python
cd "$ROOT"
mkdir -p "$DEMO/videos"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true  # the desktop wins if memory runs out
# The EB prompt sends up to ~3.9k tokens per request, so the live comparison needs a 5k context. On this 8 GB GPU
# that fits next to the two simulators without CPU offload.
MAX_LEN=${MAX_LEN:-5120}
GPU_UTIL=${GPU_UTIL:-0.76}

server_ctx() {  # the running server's context length, or nothing
  curl -s -m 2 http://127.0.0.1:8000/v1/models 2>/dev/null | $PY -c "import json,sys
try: print(json.load(sys.stdin)['data'][0]['max_model_len'])
except Exception: pass"
}

start_server() {
  local M=models/Qwen3-VL-4B-Instruct-AWQ MM='{"size": {"longest_edge": 409600, "shortest_edge": 3136}}'
  if [ "${1:-qwen3}" == "qwen25" ]; then M=models/Qwen2.5-VL-3B-Instruct-AWQ; MM='{"max_pixels": 409600}'; fi
  pgrep -f "[v]llm serve" >/dev/null && { echo "a VLM server is already running (bash demo/run_demo.sh stop)"; return 1; }
  GPU_UTIL=$GPU_UTIL MAX_LEN=$MAX_LEN MAX_SEQS=8 MM_KWARGS="$MM" EXTRA_ARGS="--mm-processor-cache-gb 0" \
    nohup bash scripts/serve_vlm.sh "$M" > "$DEMO/videos/vllm_server.log" 2>&1 &
  echo "starting $M with a $MAX_LEN-token context (log: demo/videos/vllm_server.log) ..."
  for _ in $(seq 1 90); do
    sleep 5
    [ -n "$(server_ctx)" ] && { echo "server ready"; return 0; }
    pgrep -f "[v]llm serve" >/dev/null || break
  done
  echo "server did not start; see demo/videos/vllm_server.log"; return 1
}

case "${1:-help}" in
  show)
    ctx="$(server_ctx)"
    if [ -z "$ctx" ]; then
      start_server "${2:-qwen3}" || exit 1
    elif [ "$ctx" -lt 4608 ]; then
      echo "the running VLM server has a $ctx-token context; the EB prompt needs at least 4608."
      echo "restart it with: bash demo/run_demo.sh stop && bash demo/run_demo.sh show"; exit 1
    fi
    $PY "$DEMO/live_view.py" --model "${2:-qwen3}" --open ${DEMO_ARGS:-} ;;
  replay)
    $PY "$DEMO/replay.py" --subset "${2:-long_horizon}" --episode "${3:-14}" ${DEMO_ARGS:-} ;;
  gallery)
    for ep in "long_horizon 14" "base 19" "visual_appearance 28"; do
      set -- $ep
      $PY "$DEMO/replay.py" --subset "$1" --episode "$2" ${DEMO_ARGS:-}
    done ;;
  serve)
    start_server "${2:-qwen3}" ;;
  stop)
    for p in $(pgrep -f "python[^ ]* [^ ]*live_view[.]py"); do kill "$p"; done
    sleep 2
    for p in $(pgrep -f "[v]llm serve"); do kill "$p"; done
    for p in $(pgrep -f "[t]hor-CloudRendering"); do kill "$p"; done
    echo "stopped" ;;
  *)
    sed -n 2,8p "$0" ;;
esac
