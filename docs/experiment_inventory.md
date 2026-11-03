# Experiment inventory — 2026-10-30

This is a read-only provenance index for the research assets present in `/home/ht/robotics-agent` on 2026-10-30.
It does not infer missing measurements or rerun any job.  A **result group** is one saved summary, raw JSONL
evaluation, or incomplete/legacy execution family; a queue retry is not another result group.  Formal aggregate
values below come from the named saved files; `--` means that no complete, checked aggregate exists.

## Coverage at a glance

- Research directories found: `runs/`, `logs/`, `data/`, `docs/`, and `paper/`; no `papers/` or `experiments/`
  directory exists at this revision.
- 38 independent result groups were found: 10 EB-Navigation DEV, 16 formal EB-Navigation TEST, 4 RoboTHOR,
  6 grounding JSONL evaluations, 1 legacy official smoke group, and 1 quarantined partial group.
- 7,449 raw episode-result JSON files were found: 5,577 under `runs/eb_nav/{dev,test}` (excluding copied EB logs)
  and 1,872 under `runs/robothor/` (excluding summaries).  There are 30 saved summary artifacts in these result
  trees (29 `summary.json`/`*_summary.json` plus the legacy `summary_all.json`) and 6 raw grounding JSONL artifacts.
- Named historical run versions: 12 executed versions (EB S2M DEV smoke/v2/v3/v4/v5; EB S2M TEST final r1/r2/r3;
  RoboTHOR dev v1/v2/v3/final) plus 2 frozen configuration manifests in `runs/frozen/`.  Version labels describe
  development or execution history, not independent random seeds.
- All discovered groups are assigned below to paper **body**, paper **appendix**, or **handoff**.  The paper is
  rooted at `paper/latex/main.tex`; appendix provenance is `paper/latex/sec/appendix.tex`.

## Asset map and log coverage

| Asset family | Evidence / purpose | Disposition |
|---|---|---|
| `runs/eb_nav/dev/` | S2M and oracle development episodes and summaries | appendix / handoff |
| `runs/eb_nav/test/` | formal EB-Navigation summaries, args, and per-episode records | body + appendix |
| `runs/robothor/{dev,val}/` | RoboTHOR development and full validation records | body + appendix |
| `runs/grounding/*.jsonl` | start-view grounding and candidate-verification probe outputs | appendix / handoff |
| `runs/eb_nav_official/` | six-episode legacy EB smoke record | handoff |
| `runs/quarantine/` | server-interrupted partial result records | handoff |
| `runs/frozen/*/README.md` | frozen selected-configuration notes | handoff (provenance only) |
| `logs/queue_test_main.log`, `logs/queue_from16.log`, `logs/queue_*.log` | queue controller and per-job evidence, including retries | cross-reference evidence |
| `logs/s2m_*.log`, `logs/oracle_geo_*.log`, `logs/robothor_dev_*.log` | early development and oracle diagnostics | handoff cross-reference |
| `logs/ground_*.log`, `logs/probe_verify.log`, `logs/stability_check.log` | grounding/diagnostic logs | appendix/handoff cross-reference |
| `logs/vllm_*.log`, `logs/depth_server.log`, `logs/setup_env*.log`, `logs/fetch_thor.log` | serving and environment diagnostics; not outcome evidence | handoff only |
| `data/eb_nav_dev/`, `data/robothor/` | task/evaluation configuration assets | provenance only |
| `data/depth_calib.json` | DEV depth calibration: AbsRel 0.132118, scale 0.377430, 60 frames | appendix |
| `docs/`, `paper/` | handoff, paper source, tables, bibliography and drafts | paper/handoff |

The queue controller records 26 scheduled items, not 26 statistically independent experiments.  In particular,
jobs 21--22 are retry/continuation work and are indexed as such rather than re-counted.  Some per-job logs retain
tracebacks from recoverable workers while the final summaries have complete records; those logs are preserved as
execution history, not substituted for outcome data.

## Result-group index

### EB-Navigation development and legacy groups

| Group | model / method | split, n | principal result | source evidence | state and destination |
|---|---|---:|---|---|---|
| S2M smoke | Qwen3 S2M | DEV, 4 | SR 50.0% | `runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_smoke/summary.json` | smoke; handoff |
| S2M v2 | Qwen3 S2M | DEV, 60 | Base 80.0, Long 83.3, Avg 81.7% | `runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v2/summary.json` | DEV; appendix |
| S2M v3 | Qwen3 S2M | DEV, 113 saved | incomplete; no summary | `runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v3/` | incomplete; handoff |
| S2M v4 | Qwen3 S2M | DEV, 150 | 86.7/80.0/80.0/50.0/83.3, Avg 76.0% | `runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v4/summary.json` | DEV; appendix |
| S2M v5 | Qwen3 S2M | DEV, 150 | 83.3/83.3/76.7/50.0/80.0, Avg 74.7% | `runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v5/summary.json`; `runs/frozen/s2m_v5_eb/README.md` | frozen DEV configuration; appendix/handoff |
| Oracle v1 | geometric oracle | DEV, 60 | Avg 90.0% | `runs/eb_nav/dev/oracle_geo_v1/summary.json` | privileged diagnostic; appendix |
| Oracle v2 | geometric oracle | DEV, 60 | Avg 81.7% | `runs/eb_nav/dev/oracle_geo_v2/summary.json` | privileged diagnostic; appendix |
| Oracle v3 | geometric oracle | DEV, 60 | Avg 81.7% | `runs/eb_nav/dev/oracle_geo_v3/summary.json` | privileged diagnostic; appendix |
| Oracle v4 | geometric oracle | DEV, 60 | Avg 86.7% | `runs/eb_nav/dev/oracle_geo_v4/summary.json` | privileged diagnostic; appendix |
| Oracle v5 | geometric oracle | DEV, 60 | Avg 91.7% | `runs/eb_nav/dev/oracle_geo_v5/summary.json` | privileged diagnostic; appendix |
| official legacy smoke | Qwen2.5 EB prompt | legacy, 6 | success 0; output error 2.667 | `runs/eb_nav_official/running/eb_nav/Qwen2.5-VL-3B-Instruct-AWQ_smoke/summary_all.json` | not formal; handoff |
| quarantined server-down | EB prompt | partial, 4 saved | no complete aggregate | `runs/quarantine/eb_official_r1_server_down_20261026/` | interrupted; handoff |

### Formal EB-Navigation TEST groups

Every row below is 300 episodes (five 60-episode subsets).  The five numbers are Base/Common/Complex/Visual/Long
SR; all formal summary files are in the listed directory.

| Group | model / method | result | source directory | destination |
|---|---|---|---|---|
| EB Qwen3 | Qwen3 EB prompt | 65.0/65.0/60.0/38.3/11.7; Avg 48.0; 0 JSON errors | `runs/eb_nav/test/eb_Qwen3-VL-4B-Instruct-AWQ_eb_official_r1/` | body + appendix |
| S2M r1 | Qwen3 S2M | 88.3/85.0/88.3/56.7/80.0; Avg 79.7 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1/` | body + appendix |
| S2M r2 | Qwen3 S2M | 88.3/85.0/88.3/50.0/83.3; Avg 79.0 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r2/` | body + appendix |
| S2M r3 | Qwen3 S2M | 91.7/83.3/88.3/53.3/83.3; Avg 80.0 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r3/` | body + appendix |
| EB Qwen2.5 | Qwen2.5 EB prompt | 8.3/6.7/6.7/6.7/0.0; Avg 5.7; decode errors 126/155/218/144/125 | `runs/eb_nav/test/eb_Qwen2.5-VL-3B-Instruct-AWQ_eb_official_r1/` | body caveat + appendix |
| S2M Qwen2.5 | Qwen2.5 S2M | 78.3/71.7/71.7/53.3/75.0; Avg 70.0 | `runs/eb_nav/test/s2m_Qwen2.5-VL-3B-Instruct-AWQ_final_r1/` | body + appendix |
| Oracle ceiling | geometric oracle | 93.3/93.3/93.3/93.3/83.3; Avg 91.3 | `runs/eb_nav/test/oracle_geo_none_ceiling/` | appendix diagnostic |
| no negative | Qwen3 S2M ablation | 93.3/83.3/88.3/58.3/81.7; Avg 81.0 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_negative/` | body + appendix |
| no fusion | Qwen3 S2M ablation | 78.3/70.0/75.0/41.7/80.0; Avg 69.0 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_fusion/` | body + appendix |
| no tiles | Qwen3 S2M ablation | 86.7/80.0/85.0/60.0/80.0; Avg 78.3 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_tiles/` | body + appendix |
| no zoom | Qwen3 S2M ablation | 85.0/80.0/83.3/51.7/83.3; Avg 76.7 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_zoom/` | body + appendix |
| no scan | Qwen3 S2M ablation | 95.0/91.7/91.7/71.7/48.3; Avg 79.7 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_scan/` | body + appendix |
| no exhaust | Qwen3 S2M ablation | 88.3/85.0/88.3/46.7/83.3; Avg 78.3 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_no_exhaust/` | body + appendix |
| strongest first | Qwen3 S2M ablation | 90.0/85.0/90.0/55.0/83.3; Avg 80.7 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_strongest_first/` | body + appendix |
| estimated depth | Qwen3 S2M robustness | 50.0/51.7/50.0/28.3/48.3; Avg 45.7 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_depth_est_r1/` | body + appendix |
| odometry noise | Qwen3 S2M robustness | 91.7/81.7/90.0/50.0/83.3; Avg 79.3 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_odom_noise_r1/` | body + appendix |

The Qwen3 three-run arithmetic mean is 79.6%.  Queue jobs 21--22 are continuation/retry evidence in
`logs/queue_from16.log`, not extra test runs.  Qwen2.5’s 768 JSON decoding failures affect 255/300 episodes
(Base 48, Common 54, Complex 55, Visual 50, Long 48 affected episodes); failures are planner-call counts, not
episode counts.

### RoboTHOR and grounding groups

| Group | model / method | split, n | principal result | source evidence | destination |
|---|---|---:|---|---|---|
| RoboTHOR dev v1 | Qwen3 S2M | DEV, 24 | SR/SPL 58.3/36.1 | `runs/robothor/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v1_summary.json` | appendix |
| RoboTHOR dev v2 | Qwen3 S2M | DEV, 24 | SR/SPL 75.0/41.0 | `runs/robothor/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v2_summary.json`; `runs/frozen/s2m_robothor_v2/README.md` | appendix/handoff |
| RoboTHOR dev v3 | Qwen3 S2M | DEV, 24 | SR/SPL 66.7/35.2 | `runs/robothor/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_v3_summary.json` | appendix |
| RoboTHOR final | Qwen3 S2M | VAL, 1,800 | SR 59.6, SPL 35.9, errors 17; 12 categories x150 | `runs/robothor/val/s2m_Qwen3-VL-4B-Instruct-AWQ_final_summary.json` | body + appendix |
| Qwen2.5 grounding DEV | Qwen2.5 VLM probe | DEV, 360 views | raw per-view boxes and labels | `runs/grounding/Qwen2.5-VL-3B-Instruct-AWQ_auto.jsonl` | appendix |
| Qwen2.5 grounding TEST | Qwen2.5 VLM probe | TEST, 960 views | IoU>=.3 hit/visible: 40.0/33.3/37.3/18.7% | `runs/grounding/Qwen2.5-VL-3B-Instruct-AWQ_auto_test.jsonl` | appendix |
| Qwen3 grounding DEV | Qwen3 VLM probe | DEV, 480 views | raw per-view boxes and labels | `runs/grounding/Qwen3-VL-4B-Instruct-AWQ_auto.jsonl` | appendix |
| Qwen3 grounding TEST | Qwen3 VLM probe | TEST, 960 views | IoU>=.3 hit/visible: 78.7/69.3/77.3/49.3% | `runs/grounding/Qwen3-VL-4B-Instruct-AWQ_auto_test.jsonl` | appendix |
| Qwen3 verify DEV | Qwen3 candidate verification | DEV, 105 rows | no checked aggregate | `runs/grounding/verify_Qwen3-VL-4B-Instruct-AWQ.jsonl` | handoff |
| Qwen3 verify TEST | Qwen3 candidate verification | TEST, 210 rows | no checked aggregate | `runs/grounding/verify_Qwen3-VL-4B-Instruct-AWQ_test.jsonl` | handoff |

Grounding rates use the raw saved fields: predicted box overlaps a visible ground-truth box at IoU >= 0.3,
with 75 visible target instances per TEST subset.  They are start-view probes, not end-to-end navigation success.
The verification JSONL schema is preserved but deliberately not numerically summarized without a checked metric
definition.

## Static-audit evidence

1. **Input and budget fairness.** `embagent/eb/nav_runner.py:42-49` builds RGB, depth and exact pose;
   `embagent/planners/s2m.py:77-82` consumes RGB-D/pose; `embagent/planners/geo.py:47-57` integrates depth and
   pose.  `embagent/planners/__init__.py:65-76` forwards RGB alone to EB prompt.  Target state is analysis-only
   except for the oracle (`nav_runner.py:33-49`; `embagent/planners/geo.py:144+`).  Queue/server logs show context
   ceilings, not actual consumption: Qwen3 EB/S2M 8192/2048 and Qwen2.5 EB/S2M 10240/2048.  EB records contain
   tokens; S2M token data are missing.
2. **Qwen2.5 parse failures.** `third_party/EmbodiedBench/embodiedbench/planner/nav_planner.py:235-255` calls
   `json.loads`, increments `output_json_error`, and substitutes a random action on `JSONDecodeError`.
   `embagent/planners/__init__.py:81-83` exposes the counter and `embagent/eb/nav_runner.py:195-216` aggregates it.
   `logs/queue_25.log` has 768 corresponding decode/fallback pairs.  Therefore 5.7% is materially confounded;
   schema/key failures outside this counter are unresolved.
3. **Actual S2M mechanism.** Grounding is `embagent/grounding.py:70-115`; box lifting is
   `embagent/planners/s2m.py:126-130`; map/planning is `embagent/planners/geo.py:47-117`; evidence fusion is
   `s2m.py:159-171`; choice is probability/travel-cost at `s2m.py:201-220`.  With `use_fusion=False`, persistent
   non-exhausted hypotheses are discarded and the current commitment cleared, reducing the controller to latest
   single-view evidence.  Semantic-to-metric lifting, mapping/A*, fusion, and cost-sensitive selection are
   established technique families; originality needs a separate literature review.

## Gaps and required future evidence

- No request-level S2M input/output tokens; actual context utilization and matched latency are unavailable.
- No complete summary for DEV S2M v3; only 113 episode records survive.
- No checked aggregate/metric specification for candidate-verification JSONL or the detailed failure taxonomy.
- Qwen2.5 EB score is parser-confounded; a parser-instrumented rerun is needed before any clean model comparison.
- Primary EB comparison uses unequal sensing (S2M RGB-D + pose, EB RGB).  An input-matched control and an
  untouched evaluation split are the minimum validity experiments; repeated TEST ablations create selection bias.

## Addendum 2026-10-30: offline diagnostics from saved records

`scripts/paper_diagnostics.py` (read-only; no model or simulator) regenerates every statistic added to the paper
and writes `paper/tables/diagnostics.md`.  It resolves two of the gaps above without new runs:

- **Failure taxonomy and candidate verification** now have checked aggregates (`diagnostics.md` sections 3 and 6).
  The verification probe has only 7 multi-candidate TEST frames, so it supports no method comparison.
- **Qwen2.5 EB parse failures are truncations.**  768/3,631 calls (21.2%) fail; no schema, empty-plan, or server
  failures.  Fitted completion tokens: 185 per parsed call, 4,102 per failed call (budget 4,096); failed calls are
  85.5% of completion tokens.  Raw generations were not saved.  All vLLM servers used `--generation-config vllm`,
  so the Qwen2.5 checkpoint's `repetition_penalty: 1.05` was not applied; whether it removes the truncation is
  untested.  `EB_RAW_CAPTURE_DIR` in `embagent/eb/local_model.py` (uncommitted) is intended for a DEV capture run.

Paired results now in the paper: Qwen3 S2M r1 - EB prompt +31.7 pp [+25.0, +38.3], McNemar p = 1.1e-17.  Only
no-fusion (-10.7 pp) and estimated depth (-34.0 pp) survive Holm over the nine variants; no-scan is -31.7 pp on
Long and +15.0 pp on Visual.  Candidate recall at 0.5 m on Visual is 65.0-66.7% (other subsets 93.3-98.3%); 47
of 84 Visual failures over r1-r3 never had a candidate.

### Provenance note: stale `args.json` in 12 Qwen3 S2M TEST runs (found 2026-10-30)

`args.json` in all 12 `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_*` directories records `"s2m": "{}"` and
`repeat: 0`; for depth_est_r1 / odom_noise_r1 the depth/noise flags are correct only because the retry loop passed
them.  Cause: the retry job (queues/test_main.txt line 36, logs/queue_21.log) ran `--retry-errors` without `--s2m`
/ `--repeat`, and the old `nav_runner.py` rewrote `args.json` even when 0 episodes were pending.  queue_21.log shows
"0 episodes to run" for all 12 runs, so **no episode was produced with the wrong settings**; renderer crashes had
already been absorbed by the worker's in-process retries.  The episode records confirm each switch took effect
(no_scan 0 scan actions; no_fusion at most 1 live hypothesis; no_tiles 0 tiled detections; no_negative 0 score
drops; no_zoom 0 full-frame crop queries; no_exhaust 282 vs 156 push steps).  The true flags are the ones in
queues/test_main.txt lines 18-34.  The run directories were left unmodified.  `nav_runner.py` now writes
`args.json` only when episodes run, records `VLM_EXTRA_BODY` / `VLM_MAX_TOKENS`, and refuses to resume a run
directory whose recorded result-affecting settings differ from the current flags.

### Qwen2.5 EB-prompt dev diagnosis (2026-10-30; DEV only, not reportable)

`scripts/diag_qwen25_eb.sh` (server as in the formal run: ctx 10240, util 0.66) on 2 dev episodes per subset.
Outputs: `runs/eb_nav/dev/eb_Qwen2.5-VL-3B-Instruct-AWQ_diag_{raw,rp105}/`, raw responses and truncated requests in
`runs/diag/qwen25_eb_raw{,_rp105}/`, replay in `runs/diag/qwen25_eb_raw/replay.jsonl`, log `logs/diag_qwen25_eb.log`.

- Baseline: 114 calls, 6 hit 4,096 tokens (5.3%; test was 21.2%), SR 2/10.  Every truncation is a greedy
  repetition loop: one sentence repeated inside `visual_state_description`, or `{"action_id": 0, "action_name":
  "Move forward"}` repeated without end inside `executable_plan`.  The schema bounds neither string nor array length.
- Replaying the 6 truncated requests: repetition_penalty 1.0 -> 1/6 parse; 1.05 -> 6/6 parse.
- But a live dev run with repetition_penalty 1.05 still truncates 4/85 calls (4.7%), SR 3/10: the penalty moves the
  loops to other requests rather than removing them.  **Conclusion: no rerun of the test baseline with rp 1.05**;
  the 5.7% row stands, with the truncation mechanism documented in the paper (appendix `app:qwen25`).
- Test rate 21.2% vs dev 5.3% is not explained here (dev n = 10 episodes).  A test rerun with raw capture would
  characterize it but is not needed for correctness.

### Input-matching checks for an RGB-only S2M variant (2026-10-30; DEV / offline, not reportable)

- Pose from action feedback (offline, saved TEST trajectories): failed moves give zero displacement, rotations are
  exactly +-90 deg, tilts +-30 deg; translations are snapped to the 0.1 m grid, so dead reckoning from (action,
  success) differs from the simulator pose by one cell (0.1 m) on 9.1% of steps (per-step check; cumulative drift
  not measured).  The EB-prompt baseline receives the same action/success feedback.
- `scripts/depth_scale_probe.py` on the 60 DEV calibration frames (DA-V2 Metric-Indoor-Small): global DEV scale
  AbsRel 0.132 / delta<1.25 0.818; per-frame oracle scale 0.074 / 0.920 (about half the error is per-frame scale);
  a naive floor-plane scale prior is worse (0.233 / 0.753; the bottom image band is floor-like in only 53% of frames).

### Note: server log overwritten (2026-10-30)

The bounded-baseline queue restarted vLLM through `scripts/run_queue.sh`, whose server log path depends only on
model and context, so `logs/vllm_Qwen2.5-VL-3B-Instruct-AWQ_10240.log` was overwritten.  Its uncommitted content was
the server log of the formal `eb_official_r1` Qwen2.5 run (launch 10-27 18:13).  The committed copy (git 9915cd7)
has the same launch arguments: max_model_len 10240, gpu util 0.66, max_pixels 409600, `generation_config: vllm`.
No episode record or summary was affected.  `run_queue.sh` will name server logs per queue (LOG_PREFIX) after the
running queue finishes.

### Qwen2.5 EB-prompt TEST rerun with a length-bounded schema (2026-10-30, formal)

`queues/qwen25_eb_bounded.txt` (DEV gates A/B passed, then one TEST run).  Bounds `{"max_chars": 1500,
"max_items": 20, "name_chars": 80}` fixed from DEV outputs before the TEST run; recorded in args.json.
`runs/eb_nav/test/eb_Qwen2.5-VL-3B-Instruct-AWQ_eb_bounded_r1/`, raw capture `runs/diag/qwen25_eb_bounded_test/`.
- SR 6.7/6.7/6.7/8.3/3.3, avg 6.3% [3.7, 9.3]; 2,439 calls, 0 decode failures, 0 length finishes, 0 runner errors.
- vs original schema: +0.7 pp, 15/13 discordant, McNemar p = 0.85.  S2M Qwen2.5 vs bounded EB: +63.7 pp
  [+58.0, +69.3], 192/1, p = 3.1e-56.
- 397 calls (16.3%) reach a bound exactly (361 at 20 actions, 72 at a 1,500-char field, 1 at an 80-char name;
  403 if fields of >=1,490 chars are counted): the loops remain but end at the bound.
- Qwen3 EB outputs (EB logs): longest field 823 chars, at most 6 actions, so the bounds never bind; no rerun.

### Queue/server memory changes (2026-10-30)

- `scripts/run_queue.sh`: start/resume threshold HIGH_MB 10000 -> 7000 (pause threshold LOW_MB 5000 unchanged);
  SKIP_JOBS to resume a queue; per-queue appended server logs; vLLM now starts with `--mm-processor-cache-gb 0`
  (override with VLLM_SERVER_ARGS).  The multimodal processor cache only avoids re-processing an identical image,
  so outputs are unchanged; it had grown the Qwen3 EngineCore to 6.7 GB RSS under S2M (2.5 GB without it).
  Runs before 2026-10-30 13:58 used the default cache.
- `queues/rgb_only_dev.txt` job 1 left its depth server running (1.3 GB) and blocked job 3; job 1 now stops it.

## Addendum 2026-10-30 (evening): RGB-only S2M on DEV

Queue `queues/rgb_only_dev2.txt` (unit `robotics-rgbonly`), Qwen3-VL-4B, 150 dev episodes, pose by dead reckoning
from executed actions (`--odom actions`).

| Dev config | SR |
|---|---|
| simulator depth, `--odom actions` (`gt_depth_odom_actions`) | 72.7 |
| DA-V2 Small, global scale (`rgbonly_small`) | 50.0 |

Depth, not pose, causes the drop: the 75th-percentile distance from the target to its closest detection grows from
0.19 to 0.45 m, lost-after-seen failures from 20 to 38 and approach failures from 10 to 25; the rejected-action rate
is unchanged (3.1%).

`scripts/probe_floor_scale.py` (120 dev start views, DA-V2 Base): with the single global scale 38.6% of floor
pixels within 5 m are lifted above the 0.10 m obstacle threshold. A per-frame scale that puts the floor at the known
camera height (`embagent/depth_est.floor_scale`, mode within x1.33 of the global scale) lowers this to 9.1% with
obstacle->floor errors 1.0% (global 0.7%), AbsRel 0.117 -> 0.106, median |log scale error| 0.064 -> 0.036.
Exposed as `--depth est_floor`; job 5 of the queue runs it on dev (`rgbonly_base_floor`).

DA-V2 Base, global scale (`rgbonly_base`): 52.0 on dev. Recall@1 m equals simulator depth (90.7), but 39 of its 72
failures reach a hypothesis lying within 1 m of the target and exhaust it from just outside the success radius
(simulator depth: 5 of 41); those hypotheses are 0.42 m off (simulator depth 0.09 m). Target detections are lifted
at 0.90 of the true range (simulator depth 0.97), 0.87 at 1.5-2.5 m. On the 120 probe frames, the floor scale moves
the median range ratio of furniture and objects at 1.5-4 m from 0.966/0.961 to 0.997/1.003 (IQR 0.92-1.04 -> 0.97-1.02).

DA-V2 Base + range-weighted fusion (`rgbonly_base_rw`): 49.3 on dev; the bias is systematic (detections at 0.91 of
the true range), so reweighting does not remove it; same 39 exhausted-short failures. Not adopted.

Pre-registered before it runs: `final_r4` (EB test, main config, repeat 4) is queued to measure tokens per episode on
the test set. The planner code differs from `runs/frozen/s2m_v5_eb` only by token logging and the opt-in
`pos_weight` (default reproduces the plain mean), so it is a fourth repeat and is reported with r1-r3 regardless of
its success rate.

Simulator depth + range-weighted fusion (`gtdepth_rw`, exact pose): 75.3 on dev versus 74.7 for v5 (same depth and
pose, uniform weights). Not adopted.

DA-V2 Base + per-frame floor scale (`rgbonly_base_floor`): 62.7 on dev (73.3/70.0/73.3/36.7/60.0) versus 52.0
with the global scale; exhausted-short failures 39 -> 24, rejected actions 2.5% -> 2.0%. Selected and frozen in
`runs/frozen/s2m_rgbonly_eb`. Queued on EB test after `final_r4`: `rgbonly_floor_r1`, `rgbonly_global_r1`
(ablation), `rgbonly_floor_r2`. Dev tuning of the RGB-only agent stops here.

`final_r4` (EB test, main config): 79.0 (90.0/83.3/88.3/51.7/81.7), vs r1 -0.7 pp (5/7 discordant, p=0.77). 4-run
mean 79.4 [75.0, 83.7] (3-run 79.6); paired gain over the EB prompt +31.4 [+24.8, +37.9]. Tokens per episode:
14,425 prompt / 513 completion over 52.6 calls (274 / 9.7 per call); EB prompt 28,766 / 2,425 over 8.2 calls.
Paper, tables (tab_main, supp paired and recall tables), scripts/paper_diagnostics.py, make_tables.py and
make_results_figure.py now use runs 1-4.

`rgbonly_floor_r1` (EB test, RGB only): 76.3 [71.3, 81.0] (85.0/80.0/83.3/55.0/78.3); vs EB prompt +28.3
[+21.3, +35.0], 111/26, p=1e-13; vs exact-depth final_r1 -3.3 [-7.0, +0.3], 10/20, p=0.099. Failures reaching a
true-target hypothesis and exhausting it from outside 1 m: 10 (exact depth 0; earlier Small/global/sim-pose
variant 77). Test start-view probe (240 views): floor->obstacle 42.5% (global) -> 9.9% (floor), AbsRel 0.134 ->
0.115.

`rgbonly_global_r1` (EB test, RGB only, Base, global scale): 53.0 [47.3, 58.7] (56.7/58.3/60.0/36.7/53.3); vs EB
prompt +5.0 (74/59, p=0.22; Long +41.7); floor - global +23.3 [+17.3, +29.3], 85/15, p=4.8e-13; 66 exhausted-short
failures; target range ratio 0.90 (floor 0.94). Paper: Depth paragraph rewritten around the RGB-only pair,
tab_abl sensing rows (Holm now over eleven variants; fusion 1.9e-4, crop check 0.63), tab_taxonomy, supp paired
table. Remaining: `rgbonly_floor_r2`, then headline = mean of floor r1/r2.

`rgbonly_floor_r2` (EB test): 76.7 [72.0, 81.3] (85.0/81.7/88.3/50.0/78.3); vs r1 7/6 discordant, p=1; vs EB prompt
113/27, p=1e-13. RGB-only 2-run mean 76.5 [71.8, 81.0]; vs EB prompt +28.5 [+21.7, +35.2]; vs the 4 depth-sensor
runs -2.9 [-6.3, +0.3]. Queue `queues/rgb_only_dev2.txt` finished 2026-10-31 00:03 (all jobs exit 0). Paper headline
uses the 2-run mean; tab_abl and the paired analyses use run 1 like every other variant.

## Addendum 2026-11-02: is the Qwen2.5-VL-3B EB-prompt baseline (6.3%) an artifact of our setup?

Read-only audit: call path, prompt, parser, plan execution, frames and action ids match EmbodiedBench's own code
(Qwen3-VL-4B scores 48.0 through the same path; S2M + Qwen2.5 70.0 on the same server). Two untested deviations from
how EmbodiedBench ran open models (unquantized HF checkpoints, lmdeploy/vLLM with the model's default generation
config): (1) 4-bit AWQ weights; (2) `--generation-config vllm`, which drops Qwen2.5-VL's `repetition_penalty: 1.05`
(Qwen3-VL's generation_config has none). External references: EmbodiedBench Qwen2.5-VL-7B 20.3, Qwen2-VL-7B 14.0;
SeeNav-Agent (arXiv 2512.02631, own prompt, 60 tasks) zero-shot Qwen2.5-VL-3B 16.7.

Pre-registered before any run. DEV ladder (150 episodes each, bounded schema as in eb_bounded_r1):
  D1 AWQ, greedy (current)            exp eb_bounded_dev
  D2 AWQ, repetition_penalty 1.05      exp eb_bounded_rp105_dev
  D3 FP8-Dynamic weights (RedHatAI/Qwen2.5-VL-3B-Instruct-FP8-Dynamic), repetition_penalty 1.05
Decision: the TEST baseline is rerun with the D-config of highest dev SR, ties (within 3 pp) going to the more
faithful config (D3 > D2 > D1); if that is D1, the existing eb_bounded_r1 stands. The baseline only gains from
this choice. S2M runs are not rerun. Dev numbers are diagnostics and are not reported as results.

DEV ladder results: D1 AWQ greedy 8.7 (6.7/20.0/3.3/10.0/3.3); D2 AWQ + rp 1.05 9.3; D3 FP8 + rp 1.05 at 126/150
already 44 successes (>= 29.3 whatever the remaining 24 episodes give; Base 33.3, Common 43.3, Complex 46.7, Visual
23.3). The cause is the 4-bit AWQ checkpoint of Qwen2.5-VL-3B, not decoding. Per the rule, the TEST baseline is
rerun as `eb_Qwen2.5-VL-3B-Instruct-FP8-Dynamic_eb_bounded_rp105_r1` (queued as job 5).
Extension decided now, before any Qwen3 FP8 result: the Qwen3-VL-4B EB baseline (48.0) also used AWQ weights
(third-party llm-compressor AWQ, int4 group 32), so it is rerun on TEST with Qwen's official FP8 checkpoint
(Qwen/Qwen3-VL-4B-Instruct-FP8), same schema and decoding as eb_official_r1. The paper's EB baseline rows become the
FP8 runs whatever they score; the AWQ rows move to the supplement. S2M is not rerun (user instruction), so the
paper compares S2M with 4-bit weights against the baseline with 8-bit weights, and says so.

TEST `eb_Qwen2.5-VL-3B-Instruct-FP8-Dynamic_eb_bounded_rp105_r1`: 20.3 [16.0, 25.0] (21.7/21.7/28.3/20.0/10.0),
3,156 calls, 0 decode failures, all finish=stop, completion tokens p50 183. Against the AWQ run eb_bounded_r1 (6.3):
+14.0 [+8.7, +19.3], 58/16, p=9.7e-7. S2M (AWQ, final_r1, 70.0) - EB FP8: +49.7 [+43.0, +56.3], 163/14,
p=2.3e-33; per subset +56.7/+50.0/+43.3/+33.3/+65.0. Plans at the 20-action cap 3.2% (AWQ 14.8%); action id agrees
with action name 99.6% / 95.7% for first / later plan items (AWQ 92.6% / 79.3%): the AWQ checkpoint is what produces
the loops and the id-name inconsistency. 20.3 equals EmbodiedBench's own Qwen2.5-VL-7B result.

Qwen3-VL-4B FP8 TEST (`eb_Qwen3-VL-4B-Instruct-FP8_eb_official_r1`), started 15:08: three automatic server attempts
failed because vLLM picked the DeepGEMM FP8 kernel, which needs nvcc (absent). Test starts then showed: without
DeepGEMM (CUTLASS block-scaled kernel) the 5.7 GiB weights leave 0.15 GiB of KV cache; with 1 GB offloaded to CPU
the bf16 KV cache holds 8,960 tokens, enough for the 8192 window. Running with VLLM_USE_DEEP_GEMM=0,
--cpu-offload-gb 1, --enforce-eager, util 0.84, same prompt/schema/decoding as eb_official_r1. ~78 s/episode.

## 2026-11-02: manuscript rebuilt for submission

Main text only (8 pages), no supplement. Every number in the text, both result tables and all figure data come from
`scripts/paper_assets.py` (TEST records -> `sec/numbers.tex`, `tab_main.tex`, `tab_abl.tex`,
`paper/figures/paper_data.json`); figures from `scripts/make_paper_figures.py` and `scripts/make_trajectory_figure.py`
(which also writes its caption from the records). `scripts/build_paper.sh` runs all of it and compiles. The asset
script also checks every qualitative statement of the text (e.g. "gain on every subset", "fusion is the only
significant component") and prints FAIL if a number contradicts it.
Statistics: repeated configurations enter as per-episode means; bootstrap CIs; sign-flip permutation tests
(10^6 samples, reported as p<1e-5 at the resolution limit), Holm over the ten rows of the components table; exact
McNemar for single binary runs. EB-prompt baselines use FP8 weights (S2M AWQ), stated in the setup and table caption.
Partial Qwen3 FP8 EB (118/300, 18:40): Base 66.7 (AWQ 65.0 on the same episodes), Common 65.5 (65.5).

TEST `eb_Qwen3-VL-4B-Instruct-FP8_eb_official_r1` (finished 2026-11-03 00:08): 55.0 [49.3, 60.7]
(66.7/65.0/65.0/53.3/25.0); 1 of 2,342 replies unparseable. Against its AWQ run (48.0): +7.0 [+0.7, +13.3], 58/37,
p=0.04, mostly Visual (+15.0) and Long (+13.3). S2M (4-bit, 4 runs) - EB FP8: +24.4 [+18.0, +30.8], p<1e-5; per subset
+22.9/+19.2/+23.3/-0.4/+57.1 -> S2M does not beat the 8-bit EB prompt on Visual (p=1.00); the paper says so. RGB-only:
+21.5; RGB global scale: -2.0 (p=0.65). Qwen2.5: S2M 70.0 vs EB FP8 20.3 (+49.7). Paper rebuilt with
scripts/build_paper.sh: 8-page body, all 21 claim checks pass, trajectory panels = first matching episode per subset.
