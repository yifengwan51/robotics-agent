#!/usr/bin/env bash
# Run a queue of experiments one after another, (re)starting the vLLM server with the right model and context
# size for each job. Queue file: one job per line,  MODEL_DIR | MAX_LEN | GPU_UTIL | command ...
# Lines starting with # are skipped; a job's output goes to logs/queue_<n>.log. Jobs are resumable (the runners
# skip finished episodes), so the whole queue can be restarted after an interruption.
#
# This machine is shared with the user's desktop work (IDE, browser, SIL containers). To never starve it:
#  - every process started here gets oom_score_adj 1000, so the kernel kills experiments before the user's apps;
#  - a watchdog stops the running job and the model server when MemAvailable < LOW_MB and resumes the same job
#    (resumable) only after MemAvailable has stayed above HIGH_MB for RESUME_S seconds.
#   systemd-run --user --unit=robotics-queue --property=OOMScoreAdjust=1000 --working-directory=$PWD \
#       /bin/bash -c 'bash scripts/run_queue.sh queues/test_main.txt >> logs/queue_test_main.log 2>&1'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE="${1:?queue file}"
LOW_MB="${LOW_MB:-5000}"
HIGH_MB="${HIGH_MB:-10000}"
RESUME_S="${RESUME_S:-180}"
MAX_PAUSES="${MAX_PAUSES:-30}"
cd "$ROOT"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
cur=""

log() { echo "[queue $(date +%m-%d_%H:%M:%S)] $*"; }
mem_avail_mb() { awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo; }

stop_server() {
  for p in $(pgrep -f "[v]llm serve"); do kill "$p" 2>/dev/null; done
  for _ in $(seq 1 40); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 1; done
  for p in $(pgrep -f "[v]llm serve"); do kill -9 "$p" 2>/dev/null; done
  cur=""
}

serve() {  # model_dir max_len gpu_util   (model_dir "none": job needs no model server)
  if [ "$1" == "none" ]; then stop_server; return 0; fi
  local key="$1|$2|$3"
  [ "$key" == "$cur" ] && curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
  stop_server
  local name; name="$(basename "$1")"
  local mm='{}'
  [[ "$name" == *Qwen3* ]] && mm='{"size": {"longest_edge": 409600, "shortest_edge": 3136}}'
  [[ "$name" == *2.5-VL* ]] && mm='{"max_pixels": 409600}'
  local extra_args=""
  if [[ "$name" == "Qwen3-VL-4B-Instruct-AWQ" && "$2" == "8192" ]]; then
    extra_args="--cpu-offload-gb 2"
  fi
  GPU_UTIL="$3" MAX_LEN="$2" MAX_SEQS=8 EXTRA_ARGS="$extra_args" MM_KWARGS="$mm" nohup bash scripts/serve_vlm.sh "$1" > "logs/vllm_${name}_$2.log" 2>&1 &
  for _ in $(seq 1 90); do
    sleep 5
    if curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then cur="$key"; return 0; fi
    pgrep -f "[v]llm serve" >/dev/null || break
  done
  log "vLLM failed to start for $key"; tail -3 "logs/vllm_${name}_$2.log"; return 1
}

kill_job() {  # pgid
  kill -TERM -- "-$1" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 -- "-$1" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$1" 2>/dev/null
  for p in $(pgrep -f "[t]hor-CloudRendering"); do kill -9 "$p" 2>/dev/null; done  # renderers are ours only
}

wait_for_memory() {
  local ok=0
  while [ "$ok" -lt "$RESUME_S" ]; do
    if [ "$(mem_avail_mb)" -ge "$HIGH_MB" ]; then ok=$((ok + 10)); else ok=0; fi
    sleep 10
  done
}

run_job() {  # n maxlen cmd ; returns the job's exit status, re-running it after memory pauses
  local n="$1" maxlen="$2" cmd="$3" pauses=0
  while true; do
    if ! serve "$model" "$maxlen" "$util"; then return 99; fi
    VLM_MAX_MODEL_LEN="$maxlen" setsid bash -c "$cmd" >> "logs/queue_${n}.log" 2>&1 &
    local pid=$! low=0 dead=0
    while kill -0 "$pid" 2>/dev/null; do
      if [ "$(mem_avail_mb)" -lt "$LOW_MB" ]; then low=1; break; fi
      if [ "$model" != "none" ] && ! pgrep -f "[v]llm serve" >/dev/null; then dead=1; break; fi
      sleep 3
    done
    if [ "$dead" -eq 1 ]; then  # model server crashed under the job: stop it, restart the server, resume the job
      pauses=$((pauses + 1))
      log "job $n: model server DIED (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); restarting ($pauses/$MAX_PAUSES)"
      kill_job "$pid"; cur=""
      [ "$pauses" -ge "$MAX_PAUSES" ] && { log "job $n gave up after $pauses restarts"; return 97; }
      sleep 20
      continue
    fi
    if [ "$low" -eq 0 ]; then wait "$pid"; return $?; fi
    pauses=$((pauses + 1))
    log "job $n PAUSED: MemAvailable $(mem_avail_mb) MB < $LOW_MB MB (pause $pauses/$MAX_PAUSES); stopping job and model server"
    kill_job "$pid"
    stop_server
    [ "$pauses" -ge "$MAX_PAUSES" ] && { log "job $n gave up after $pauses pauses"; return 98; }
    wait_for_memory
    log "job $n RESUMING (MemAvailable $(mem_avail_mb) MB)"
  done
}

n=15
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  n=$((n + 1))
  IFS='|' read -r model maxlen util cmd <<< "$line"
  model="$(echo "$model" | xargs)"; maxlen="$(echo "$maxlen" | xargs)"; util="$(echo "$util" | xargs)"
  log "job $n: $(basename "$model") ctx=$maxlen util=$util :: $cmd"
  if [ "$(mem_avail_mb)" -lt "$HIGH_MB" ]; then log "waiting for memory before job $n"; wait_for_memory; fi
  touch "logs/queue_${n}.log"
  start=$(wc -l < "logs/queue_${n}.log" 2>/dev/null || echo 0)
  run_job "$n" "$maxlen" "$cmd"
  rc=$?
  log "job $n exit $rc :: $(tail -n +"$((start + 1))" "logs/queue_${n}.log" 2>/dev/null | grep -E "AVG\(subsets\)|'type': 'ALL'" | tail -1)"
  if [ "$rc" -ne 0 ]; then
    log "FATAL: job $n failed (exit $rc); stopping queue"
    stop_server
    exit "$rc"
  fi
done < "$QUEUE"
stop_server
log "all jobs done"
