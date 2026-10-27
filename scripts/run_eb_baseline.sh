#!/usr/bin/env bash
# EB official-prompt baseline on the EB-Navigation TEST split (nothing is tuned on it; prompts are EB's own).
# Long prompts (~3.5k tokens + EB's 4096-token completion budget) need the 8k-context server, which leaves room
# for one AI2-THOR renderer only, so this runs with a single worker.
#   bash scripts/run_eb_baseline.sh <model_dir> <exp_name> [extra nav_runner args...]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${1:?model dir}"; EXP="${2:?exp name}"; shift 2
NAME="$(basename "$MODEL_DIR")"
cd "$ROOT"

for p in $(pgrep -f "[v]llm serve"); do kill "$p"; done
for _ in $(seq 1 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 1; done

MM='{}'
[[ "$NAME" == *Qwen3* ]] && MM='{"size": {"longest_edge": 409600, "shortest_edge": 3136}}'
GPU_UTIL="${GPU_UTIL:-0.80}" MAX_LEN=8192 MM_KWARGS="$MM" \
  nohup bash scripts/serve_vlm.sh "$MODEL_DIR" > "logs/vllm_${NAME}_8k.log" 2>&1 &
for _ in $(seq 1 90); do
  sleep 5
  curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  pgrep -f "[v]llm serve" >/dev/null || { echo "vLLM failed to start"; tail -5 "logs/vllm_${NAME}_8k.log"; exit 1; }
done
echo "server up: $NAME (8k context)"

VLM_MAX_MODEL_LEN=8192 VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-4096}" ~/miniforge3/envs/embnav_thor/bin/python -m embagent.eb.nav_runner \
  --planner eb --model "$NAME" --exp "$EXP" --workers 1 "$@"
