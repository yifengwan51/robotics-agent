#!/usr/bin/env bash
# (Re)start the CPU monocular-depth server with a given checkpoint under models/ and wait until it reports it.
#   bash scripts/depth_server.sh DA2-Metric-Indoor-Base      (port $DEPTH_PORT, default 8010)
#   bash scripts/depth_server.sh stop
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${DEPTH_PORT:-8010}"
for p in $(pgrep -f "[e]mbagent.depth_est --serve"); do kill "$p" 2>/dev/null; done
for _ in $(seq 1 20); do pgrep -f "[e]mbagent.depth_est --serve" >/dev/null || break; sleep 1; done
[ "${1:?model name or stop}" == "stop" ] && exit 0
cd "$ROOT"
CUDA_VISIBLE_DEVICES= DEPTH_MODEL="$1" DEPTH_PORT="$PORT" nohup bash -c \
  'echo 1000 > /proc/self/oom_score_adj; exec ~/miniforge3/envs/vlm/bin/python -m embagent.depth_est --serve --device cpu --threads 8' \
  > "logs/depth_server_$1.log" 2>&1 &
for _ in $(seq 1 60); do
  sleep 2
  [ "$(curl -s -m 2 "http://127.0.0.1:$PORT/" 2>/dev/null)" == "$1" ] && { echo "depth server $1 on :$PORT"; exit 0; }
done
echo "depth server $1 did not start"; tail -3 "logs/depth_server_$1.log"; exit 1
