#!/usr/bin/env bash
# Waits for both Qwen3-VL-2B checkpoints, then runs queues/qwen3_2b_test.txt (S2M AWQ, then the FP8 EB prompt).
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet robotics-dl2b; do sleep 30; done
for m in models/Qwen3-VL-2B-Instruct-AWQ models/Qwen3-VL-2B-Instruct-FP8; do
  ls "$m"/*.safetensors >/dev/null 2>&1 || { echo "$(date '+%F %T') missing weights in $m"; exit 1; }
done
VLLM_USE_DEEP_GEMM=0 LOG_PREFIX=q3_2b bash scripts/run_queue.sh queues/qwen3_2b_test.txt
