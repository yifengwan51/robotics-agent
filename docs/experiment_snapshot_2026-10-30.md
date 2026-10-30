# Experiment snapshot — 2026-10-30

Scope: static handoff snapshot only. No experiment, simulator, model server, algorithm, or evaluation configuration was changed.

## Queue completion and artifact index

`logs/queue_from16.log` records jobs 7--26 and ends with `all jobs done`; jobs 1--6 are recorded in
`logs/queue_test_main.log` / its retained `prev*` logs. The 26-item queue is complete, but the queue log is a
controller record rather than a substitute for the saved `summary.json` artifacts.

| Jobs | Work | Primary evidence |
|---|---|---|
| 1--2 | RoboTHOR development smoke runs | `logs/queue_test_main.log`; `runs/robothor/dev/*_summary.json` |
| 3 | Qwen3 EB-prompt baseline | `runs/eb_nav/test/eb_Qwen3-VL-4B-Instruct-AWQ_eb_official_r1/summary.json` |
| 4 | Oracle geometry ceiling | `runs/eb_nav/test/oracle_geo_none_ceiling/summary.json` |
| 5, 10, 23 | Grounding probes | `runs/grounding/*.jsonl` |
| 6, 11 | Verification probes | `logs/queue_from16.log`; probe outputs require separate review before reporting |
| 7--9 | Qwen3 S2M test runs 1--3 | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r{1,2,3}/summary.json` |
| 12--13 | Estimated depth; odometry noise | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_{depth_est_r1,odom_noise_r1}/summary.json` |
| 14--20 | Seven S2M component checks | `runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_abl_*/summary.json` |
| 21--22 | Retry pass for S2M and Qwen3 EB | `logs/queue_from16.log`; existing completed episode records were retained |
| 24--25 | Qwen2.5 S2M and EB-prompt runs | `runs/eb_nav/test/{s2m,eb}_Qwen2.5-VL-3B-Instruct-AWQ_*/*` |
| 26 | RoboTHOR official validation | `runs/robothor/val/s2m_Qwen3-VL-4B-Instruct-AWQ_final_summary.json`; `logs/queue_26.log` |

## Verified metrics

All EB-Navigation formal result directories contain five 60-episode subsets (300 total).

- Qwen3-VL-4B S2M: 79.7, 79.0, 80.0% over runs 1--3; mean 79.6%.
- Qwen3-VL-4B EB prompt: 48.0% (300 episodes; no recorded JSON decoding errors).
- Qwen2.5-VL-3B S2M: 70.0% (300 episodes).
- Qwen2.5-VL-3B EB prompt: 5.7% (300 episodes). The saved episode records total 768 JSON decode failures across 255 affected episodes (base 126/48, common 155/54, complex 218/55, visual 144/50, long 125/48; pairs are failures/affected episodes).
- Qwen3 ablations: no-negative 81.0%; no-fusion 69.0%; no-tiles 78.3%; no-zoom/crop-verification 76.7%; no-scan/look-around 79.7%; no-exhaust 78.3%; strongest-first 80.7%. Each is one 300-episode test run.
- Robustness: estimated depth 45.7%; 5% translational + 2-degree rotational odometry noise 79.3%. Both are one 300-episode test run.
- RoboTHOR validation: 1800 episodes, SR 59.6%, SPL 35.9%, 17 logged errors.

The LaTeX paper is rooted at `paper/latex/main.tex`. This handoff updated `sec/0_abstract.tex`, `sec/5_experiments.tex`,
`sec/6_conclusion.tex`, `sec/tables.tex`, and the bodies in `paper/tables/`.

## Static review findings

### A. Inputs and context are not matched

`embagent/eb/nav_runner.py:42-49` constructs RGB, simulator depth, and exact pose. S2M reads all three
(`embagent/planners/s2m.py:77-82`) and turns depth + pose into a map (`embagent/planners/geo.py:47-57`). The
EB-prompt adapter saves and forwards only `obs["rgb"]` (`embagent/planners/__init__.py:65-76`); it receives action
feedback but no depth, pose, odometry, or map. `nav_runner.py:33-49` shows target position/state is logged only
for analysis; only the diagnostic oracle planner receives `oracle_target`.

Conclusion: the primary S2M versus EB-prompt difference is input-unequal (RGB-D + exact pose versus RGB), so it
cannot be called an algorithm-only gain. Qwen3 EB 8192 / S2M 2048 and Qwen2.5 EB 10240 / S2M 2048 are vLLM
configuration limits documented in `logs/queue_from16.log`, not actual token use. EB episode records log actual
mean prompt/completion tokens; S2M does not log token counts.

### B. Qwen2.5 JSON failures are a material confound

`third_party/EmbodiedBench/embodiedbench/planner/nav_planner.py:235-255` calls `json.loads`; on a
`JSONDecodeError` it increments `output_json_error` and substitutes a random action. The project adapter exposes
that counter in `embagent/planners/__init__.py:81-83`, and `embagent/eb/nav_runner.py:195-216` sums it into each
summary. `logs/queue_25.log` contains 768 matching `Failed to decode JSON` / `random action` pairs.

Conclusion: 768 is the number of failed planner calls, not the number of failed episodes; 255/300 episodes had at
least one. The 5.7% score can be substantially distorted by parser fallback. It should not be used to quantify a
clean Qwen2.5 algorithmic advantage. Unresolved: counters do not separately label schema/key failures handled by
the broad exception branch, so the saved count is a lower-level decode counter, not a complete output-quality audit.

### C. What S2M implements

Target semantics are queried in `embagent/grounding.py:70-115`; boxes are lifted with depth and pose in
`embagent/planners/s2m.py:126-130`; the depth map and geometric path planner are in
`embagent/planners/geo.py:47-117`. Multi-view fusion is `s2m.py:159-171`: with `use_fusion=True`, nearby 3-D
detections update a persistent hypothesis; with `use_fusion=False`, the implementation removes non-exhausted
hypotheses and clears the current commitment, i.e., it trusts the latest single-view detection. Candidate selection
uses the probability-over-travel-cost index in `s2m.py:201-220`; tiled/crop verification is `s2m.py:132-153`.

Conclusion: semantic-to-metric lifting, occupancy mapping/A*, evidence fusion, and cost-sensitive candidate
selection are established families of techniques. The artifact supports the concrete integration and ablation
observations, but does not establish novelty from module names. A literature review is required to identify any
defensible original contribution claim.

## Three key unresolved issues

1. Build an input-matched baseline: either RGB-only S2M or an EB prompt/controller given the same depth and pose.
2. Repair or constrain the Qwen2.5 JSON-output evaluation and rerun its EB-prompt baseline; separately log decode,
schema, and episode-level fallback impact.
3. Freeze model selection on development data and perform an independent held-out evaluation: the test set has
already been used repeatedly for component comparisons, creating test-set selection bias.

## Minimum future experiments (priority order)

1. One input-matched Qwen3 baseline/control on a fresh held-out split, with per-episode paired outputs.
2. One Qwen2.5 EB-prompt rerun after parser-output instrumentation, reporting failures and affected episodes.
3. One independent final verification run after choosing the final configuration on development data only.
4. If an efficiency claim is retained, log request-level input/output tokens and wall time for both planners under
the same server and rendering settings.

## Claude review order

1. `paper/latex/sec/5_experiments.tex` and `paper/tables/main_eb_nav.tex`
2. `embagent/eb/nav_runner.py`, `embagent/planners/__init__.py`, and `embagent/planners/s2m.py`
3. `third_party/EmbodiedBench/embodiedbench/planner/nav_planner.py`
4. `runs/eb_nav/test/*/summary.json` and `logs/queue_from16.log`
5. `runs/robothor/val/s2m_Qwen3-VL-4B-Instruct-AWQ_final_summary.json`
