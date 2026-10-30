#!/usr/bin/env bash
# Diagnose the Qwen2.5-VL-3B EB-prompt JSON failures on the DEV split only (no test episodes).
#   1. serve Qwen2.5 exactly as the formal run did (ctx 10240, util 0.66, max_pixels 409600)
#   2. EB-prompt planner on 2 dev episodes per subset with raw-response capture         -> diag_raw
#   3. replay every truncated request with and without repetition_penalty 1.05
#   4. same 10 dev episodes with repetition_penalty 1.05                                 -> diag_rp105
# Launch:  systemd-run --user --unit=robotics-diag --property=OOMScoreAdjust=1000 --working-directory=$PWD \
#              /bin/bash -c 'bash scripts/diag_qwen25_eb.sh >> logs/diag_qwen25_eb.log 2>&1'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
PY=~/miniforge3/envs/embnav_thor/bin/python
MODEL=models/Qwen2.5-VL-3B-Instruct-AWQ
NAME=Qwen2.5-VL-3B-Instruct-AWQ
CAP=runs/diag/qwen25_eb_raw
log() { echo "[diag $(date +%m-%d_%H:%M:%S)] $*"; }

if ! curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q "$NAME"; then
  pgrep -f "[v]llm serve" >/dev/null && { log "another vLLM server is running; aborting"; exit 1; }
  GPU_UTIL=0.66 MAX_LEN=10240 MAX_SEQS=8 MM_KWARGS='{"max_pixels": 409600}' \
    nohup bash scripts/serve_vlm.sh "$MODEL" > logs/vllm_diag_qwen25.log 2>&1 &
  for _ in $(seq 1 90); do sleep 5; curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; done
fi
curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q "$NAME" || { log "vLLM did not start"; exit 1; }
log "server up"

log "step 2: dev capture run"
EB_RAW_CAPTURE_DIR="$CAP" VLM_MAX_MODEL_LEN=10240 $PY -m embagent.eb.nav_runner --planner eb --model "$NAME" \
  --exp diag_raw --data data/eb_nav_dev --limit 2 --workers 1
log "step 3: replay truncated requests"
~/miniforge3/envs/vlm/bin/python scripts/replay_truncated.py "$CAP"
log "step 4: dev run with repetition_penalty 1.05"
EB_RAW_CAPTURE_DIR="${CAP}_rp105" VLM_EXTRA_BODY='{"repetition_penalty": 1.05}' VLM_MAX_MODEL_LEN=10240 \
  $PY -m embagent.eb.nav_runner --planner eb --model "$NAME" --exp diag_rp105 --data data/eb_nav_dev --limit 2 \
  --workers 1
log "done"
for p in $(pgrep -f "[v]llm serve"); do kill "$p" 2>/dev/null; done
