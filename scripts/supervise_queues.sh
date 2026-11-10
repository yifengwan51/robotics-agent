#!/usr/bin/env bash
# Run queues in order and restart a queue that stops on a failed job (runners are resumable, finished episodes are
# skipped). A queue is given up after MAX_TRIES failures; the status line goes to logs/supervisor.log.
#   bash scripts/supervise_queues.sh PREFIX:QUEUE_FILE [PREFIX:QUEUE_FILE ...]
set -uo pipefail
cd "$(dirname "$0")/.."
MAX_TRIES="${MAX_TRIES:-4}"
log() { echo "[supervisor $(date +%m-%d_%H:%M:%S)] $*" >> logs/supervisor.log; }
for item in "$@"; do
  prefix="${item%%:*}"; queue="${item#*:}"
  for try in $(seq 1 "$MAX_TRIES"); do
    log "start $queue (try $try)"
    LOG_PREFIX="$prefix" bash scripts/run_queue.sh "$queue" >> "logs/queue_$prefix.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then log "DONE $queue"; break; fi
    log "queue $queue exited $rc; restarting in 120 s"
    sleep 120
    [ "$try" -eq "$MAX_TRIES" ] && log "GAVE UP $queue after $MAX_TRIES tries"
  done
done
log "ALL QUEUES FINISHED"
