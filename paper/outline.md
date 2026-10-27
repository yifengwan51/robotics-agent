# Paper outline (target: ICCV 2027; backup ACM MM 2027 / NeurIPS 2027)

Working title: **Small VLMs Know What, Not Where: Semantic-to-Metric Grounding for Embodied Navigation with Local Models**

Numbers marked [D] are dev-split results (for development only); every number in the paper will come from the
official EB-Navigation test split, run once per frozen config with 3 repeats.

## 1. Introduction
- Embodied agents built on frontier VLMs work; small local VLMs (<=8B, what fits on a robot or an 8 GB GPU) fail
  badly on EB-Navigation (14-45 SR; 0-17 on Long-horizon). Prior training-free gains are reported only for >=9B
  or API models (EMHO, BrainMem, ETP); SeeNav's large gains rely on a simulator-rendered BEV with GT target boxes.
- Question: *why* do small VLMs fail? Diagnosis: they recognise the target reasonably well
  (per-frame hit rate, 3D error 0.18 m when hit [D]) but cannot turn recognition into metric action
  (spatial reasoning / action errors dominate). Once "where" is known, navigation is nearly solved
  (oracle-target geometry 91.7 [D]).
- Method: S2M restricts the VLM to semantic questions and hands every metric decision to geometry; evidence is
  fused across distinct views in 3D, with negative evidence, hysteresis and an uncertainty-triggered magnified search.
- Results: (to fill) 4B local model reaches X on EB-Nav test (vs Y same-model baseline, Z GPT-4o), largest gain on
  Long-horizon; holds across 3 VLM families, with estimated depth and noisy odometry; second benchmark RoboTHOR.

Contributions
1. First quantified failure analysis of <=8B VLM agents on EB-Navigation: separates *recognition* from *metric
   grounding* via per-frame grounding probes and oracle ceilings.
2. S2M, a training-free semantic-to-metric agent: VLM for "what/whether/where in the image", geometry for
   "where in the world / how to get there"; multi-view evidence fusion with negative evidence and hysteresis;
   magnified tiled search with crop verification for small, distant targets.
3. A strict evaluation protocol: held-out dev split built with the benchmark's own generator, no privileged state,
   3 repeats, bootstrap CIs and paired tests; released code for local 8 GB GPUs.

## 2. Related work
- VLM/MLLM embodied agents and benchmarks: EmbodiedBench, VAGEN, ERA, Embodied-Reasoner, SeeNav, EMHO, ETP, BrainMem.
- Zero-shot object navigation with foundation models: CoW, ESC, VLFM, SG-Nav, UniGoal, VLMnav, CogNav, WMNav.
- Visual prompting / grounding for action: PIVOT, Set-of-Mark, RoboPoint; test-time zoom (V*, ZoomEye).
- Uncertainty and verification for VLM agents: KnowNo, Explore-until-Confident, verifier-guided selection.

## 3. Diagnosis (Sec. "Why small VLMs fail")
- Table: per-frame grounding (presence AUC, hit@IoU0.3, boxes-on-invisible, 3D lift error) per model per subset.
- Table: oracle ceilings — oracle target + our geometry; oracle box + our geometry; VLM end-to-end.
- Fig: failure taxonomy of the EB prompt baseline (never-seen / lost / approach-fail / blocked / oscillation).

## 4. Method
- 4.1 Semantic queries (presence logprob, box, target spec from the instruction).
- 4.2 Metric lifting and map (depth back-projection, occupancy, pose-space A* with "no blind steps").
- 4.3 Evidence fusion (distinct-view weighting, negative evidence with occlusion test, commit/keep hysteresis,
  nearest-first, exhaust-on-arrival).
- 4.4 Uncertainty-triggered magnified search (tiles + crop verification).
- Fig 1 (teaser): baseline vs S2M trajectory + hypothesis map.

## 5. Experiments
- Setup: EB-Navigation 5 subsets x 60; models Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3.5-4B (AWQ, vLLM, RTX 5060 8 GB).
- Baselines: EB official prompt, EB + chat history, published numbers (GPT-4o, InternVL3-8B, Qwen2.5-VL-7B, EMHO).
- Main table: SR per subset + avg, mean +- std over 3 repeats, 95% CI.
- Second benchmark: RoboTHOR ObjectNav val (SR/SPL) vs SG-Nav-LLaMA-7B, CoW.
- Ablations: -negative evidence, -fusion, -hysteresis/nearest, -tiles, -zoom, -scan, -exhaust (important: shows the
  method does not rely on EB's auto-termination).
- Robustness: GT depth vs Depth-Anything-V2 metric (calibrated on dev); odometry noise levels.
- Efficiency: VLM calls/tokens/latency per episode vs the EB prompt baseline.
- Qualitative: MuJoCo LTR M1 wheel-legged demo (optional).

## 6. Limitations
- GT depth / odometry assumptions (mitigated by robustness study); EB auto-terminates on success (exhaust ablation +
  RoboTHOR with explicit STOP); same-category multi-instance ambiguity; single-room scenes.
