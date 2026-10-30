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
