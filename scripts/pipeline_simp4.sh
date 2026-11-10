#!/usr/bin/env bash
# dev ablations -> pre-registered gate -> freeze code -> TEST queue. Status in logs/pipeline_simp4.log.
cd "$(dirname "$0")/.."
L=logs/pipeline_simp4.log
log() { echo "[pipeline $(date +%m-%d_%H:%M:%S)] $*" >> $L; }
while systemctl --user is-active --quiet robotics-supervisor5; do sleep 60; done
log "dev ablations finished"
if ! ~/miniforge3/envs/embnav/bin/python scripts/gate_simp4.py >> $L 2>&1; then log "PIPELINE END: GATE FAILED, test not started"; exit 1; fi
log "gate passed; freezing"
F=runs/frozen/s2m_simp4_eb; mkdir -p $F
rsync -a --exclude __pycache__ embagent $F/
( cd $F && find embagent -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
cat > $F/README.md <<TXT
Frozen simplified S2M for the EB-Navigation TEST runs (frozen $(date +%F) after the dev gate, scripts/gate_simp4.py).
Planner kwargs: {"scan_mode": "gated", "use_negative": false, "use_tiles": false, "nearest_first": false}
Modules: multi-view evidence fusion, look-around gated on an empty view, exhaustion on arrival, crop check.
Selected on the 450-episode dev set (data/eb_nav_dev, _s1, _s2: 30 rooms not in the test set). Dev diagnosis
started from the v5 test ablations, in which negative evidence, tiled search and the search index showed no effect.
TXT
log "frozen to $F; starting TEST queue"
: > logs/supervisor.log
bash scripts/supervise_queues.sh simp4test:queues/simp4_test.txt
log "PIPELINE END: $(tail -1 logs/supervisor.log)"
