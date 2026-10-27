#!/usr/bin/env bash
# Serve a local VLM with vLLM (OpenAI-compatible API on 127.0.0.1:$PORT), sized to share the 8 GB GPU
# with the AI2-THOR renderer.
#   bash scripts/serve_vlm.sh [model_dir] [served_name]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/Qwen2.5-VL-3B-Instruct-AWQ}"
NAME="${2:-$(basename "$MODEL")}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.62}"
# Qwen2.5-VL takes max_pixels; other processors use their own defaults (500x500 frames are small anyway)
if [ -z "${MM_KWARGS:-}" ]; then
  if [[ "$NAME" == *2.5-VL* ]]; then MM_KWARGS='{"max_pixels": 409600}'; else MM_KWARGS='{}'; fi
fi
MAX_LEN="${MAX_LEN:-10240}"  # EB-Nav prompts ~3.5k tokens + EB's 4096-token completion budget

# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlm
# FlashInfer's sampling kernels need nvcc for JIT; the PyTorch sampler is fine (we decode greedily).
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation on the shared 8 GB GPU
exec vllm serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" --max-num-seqs "${MAX_SEQS:-8}" \
  --limit-mm-per-prompt '{"image": 2}' --mm-processor-kwargs "$MM_KWARGS" \
  --max-num-batched-tokens "${MAX_BATCHED:-2048}" \
  --generation-config vllm ${EXTRA_ARGS:-}
