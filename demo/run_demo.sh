#!/usr/bin/env bash
# One entry point for the demo videos. Run from anywhere:
#   bash demo/run_demo.sh gallery                      three replayed test episodes (no model needed)
#   bash demo/run_demo.sh replay long_horizon 14 both  one replayed episode: agent = both | s2m | eb
#   bash demo/run_demo.sh show [qwen3|qwen25]          live browser view (starts the VLM server if needed)
#   bash demo/run_demo.sh serve [qwen3|qwen25]         start the local VLM server (needed for live / show)
#   bash demo/run_demo.sh live base 19 [test|dev]      run S2M live on one episode and record it
#   bash demo/run_demo.sh stop                         stop the VLM server and any simulator left running
# Extra options go to the Python scripts through DEMO_ARGS, e.g. DEMO_ARGS="--lang en --hold 1.0".
set -uo pipefail
DEMO="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DEMO")"
PY=~/miniforge3/envs/embnav_thor/bin/python
cd "$ROOT"
mkdir -p "$DEMO/videos"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true  # the desktop wins if memory runs out

start_server() {  # model: qwen3 | qwen25
  case "$1" in
    qwen3) M=models/Qwen3-VL-4B-Instruct-AWQ; MM='{"size": {"longest_edge": 409600, "shortest_edge": 3136}}' ;;
    qwen25) M=models/Qwen2.5-VL-3B-Instruct-AWQ; MM='{"max_pixels": 409600}' ;;
    *) echo "unknown model $1"; return 1 ;;
  esac
  pgrep -f "[v]llm serve" >/dev/null && { echo "a VLM server is already running (bash demo/run_demo.sh stop)"; return 1; }
  GPU_UTIL=0.68 MAX_LEN=2048 MAX_SEQS=8 MM_KWARGS="$MM" EXTRA_ARGS="--mm-processor-cache-gb 0" \
    nohup bash scripts/serve_vlm.sh "$M" > "$DEMO/videos/vllm_server.log" 2>&1 &
  echo "starting $M (log: demo/videos/vllm_server.log) ..."
  for _ in $(seq 1 90); do
    sleep 5
    curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "server ready"; return 0; }
    pgrep -f "[v]llm serve" >/dev/null || break
  done
  echo "server did not start; see demo/videos/vllm_server.log"; return 1
}

case "${1:-help}" in
  gallery)
    for ep in "long_horizon 14" "base 19" "visual_appearance 28"; do
      set -- $ep
      $PY "$DEMO/replay.py" --subset "$1" --episode "$2" --agent both ${DEMO_ARGS:-}
    done ;;
  replay)
    $PY "$DEMO/replay.py" --subset "${2:-long_horizon}" --episode "${3:-14}" --agent "${4:-both}" ${DEMO_ARGS:-} ;;
  serve)
    start_server "${2:-qwen3}" ;;
  show)
    # live browser view: starts the VLM server if needed, then serves http://127.0.0.1:8765
    curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || start_server "${2:-qwen3}" || exit 1
    $PY "$DEMO/live_view.py" --model "${2:-qwen3}" --open ${DEMO_ARGS:-} ;;
  live)
    $PY "$DEMO/live.py" --subset "${2:-base}" --episode "${3:-19}" --split "${4:-test}" \
      --preview "$DEMO/videos/preview.png" ${DEMO_ARGS:-} ;;
  stop)
    for p in $(pgrep -f "python[^ ]* [^ ]*live_view[.]py"); do kill "$p"; done
    for p in $(pgrep -f "[v]llm serve"); do kill "$p"; done
    for p in $(pgrep -f "[t]hor-CloudRendering"); do kill "$p"; done
    echo "stopped" ;;
  *)
    sed -n 2,9p "$0" ;;
esac
