#!/usr/bin/env bash
# 2026-11-23 extra experiments, in order: the TEST sensitivity study (queues/sens_test.txt), which runs while the
# Qwen3-VL-2B weights download, then Qwen3-VL-2B on EB TEST (queues/qwen3_2b_test.txt).
# The desktop shares the 8 GB GPU and 32 GB RAM, so a queue can stop when a model server finds too little GPU memory
# after a memory pause. Runners skip finished episodes, so each queue is retried after the GPU is free again.
set -uo pipefail
cd "$(dirname "$0")/.."
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() {
  local ok=0
  while [ "$ok" -lt 3 ]; do  # free for three checks in a row, so a server that is still shutting down is not counted
    if [ "$(gpu_free)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; echo "$(date '+%T') waiting for GPU: $(gpu_free) MiB free"; fi
    sleep 20
  done
}
run_q() {  # queue, log prefix, extra env
  for attempt in $(seq 1 12); do
    wait_gpu 5800
    env "${@:3}" LOG_PREFIX="$2" bash scripts/run_queue.sh "$1" && return 0
    echo "$(date '+%F %T') queue $1 stopped (attempt $attempt); retrying"
    for p in $(pgrep -f "[v]llm serve|VLLM::EngineCore"); do kill "$p" 2>/dev/null; done
    sleep 60
    for p in $(pgrep -f "[v]llm serve|VLLM::EngineCore"); do kill -9 "$p" 2>/dev/null; done
  done
  return 1
}
run_q queues/sens_test.txt sens
while systemctl --user is-active --quiet robotics-dl2b; do sleep 30; done
if ls models/Qwen3-VL-2B-Instruct-AWQ/*.safetensors models/Qwen3-VL-2B-Instruct-FP8/*.safetensors >/dev/null 2>&1; then
  run_q queues/qwen3_2b_test.txt q3_2b VLLM_USE_DEEP_GEMM=0
else
  echo "$(date '+%F %T') Qwen3-VL-2B weights missing; skipping the 2B runs"
fi
