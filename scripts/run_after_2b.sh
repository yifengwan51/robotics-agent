#!/usr/bin/env bash
# 2026-11-23 extra experiments, in order: the TEST sensitivity study (queues/sens_test.txt), which runs while the
# Qwen3-VL-2B weights download, then Qwen3-VL-2B on EB TEST (queues/qwen3_2b_test.txt). Waits for the desktop to leave
# enough GPU memory free.
set -uo pipefail
cd "$(dirname "$0")/.."
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { until [ "$(gpu_free)" -ge "$1" ]; do echo "$(date '+%T') waiting for GPU: $(gpu_free) MiB free"; sleep 60; done; }
wait_gpu 5800
LOG_PREFIX=sens bash scripts/run_queue.sh queues/sens_test.txt
while systemctl --user is-active --quiet robotics-dl2b; do sleep 30; done
if ls models/Qwen3-VL-2B-Instruct-AWQ/*.safetensors models/Qwen3-VL-2B-Instruct-FP8/*.safetensors >/dev/null 2>&1; then
  wait_gpu 5800
  VLLM_USE_DEEP_GEMM=0 LOG_PREFIX=q3_2b bash scripts/run_queue.sh queues/qwen3_2b_test.txt
else
  echo "$(date '+%F %T') Qwen3-VL-2B weights missing; skipping the 2B runs"
fi
