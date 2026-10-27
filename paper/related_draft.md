# Related work (draft; every citation is in paper/verified_refs.tsv — add nothing that is not verified)

**MLLM agents for embodied tasks.**
- EmbodiedBench (ICML 2025) evaluates MLLM agents on high-level planning (EB-ALFRED, EB-Habitat) and low-level
  control (EB-Navigation, EB-Manipulation). It finds a large gap between frontier models and <=8B open models,
  especially when the target is initially out of view.
- Training-based approaches close the gap by fine-tuning small VLMs on embodied data or with RL: ERA, the RL
  variant of SeeNav-Agent, Embodied-Reasoner. This needs task-specific trajectories and GPU clusters.
- Training-free approaches add memory (BrainMem), tools (ETP), visual prompts (SeeNav-Agent) or an automatically
  searched harness (EMHO). Their reported gains use 9B-27B or API models. SeeNav's visual prompt is drawn on a
  simulator-rendered bird's-eye view with the target box.
- S2M is training-free, uses no privileged rendering, and targets 3-4B local models. It differs from tool- and
  harness-based agents in *where* the VLM is used: the VLM answers only per-image semantic questions and never
  chooses motions.

**Zero-shot object navigation with foundation models.**
- Modular ObjectNav agents combine open-vocabulary perception with maps and planners: CoW (CVPR 2023), VLFM
  (ICRA 2024), SG-Nav (NeurIPS 2024), UniGoal (CVPR 2025), CogNav (ICCV 2025). Most rely on separate detectors
  and an LLM/VLM for frontier or scene-graph reasoning, and are evaluated on HM3D/MP3D.
- VLMnav (arXiv 2024) lets a VLM pick among depth-derived navigable directions.
- S2M shares the semantic-map/planner structure but is built for *small* VLMs. One local model does recognition,
  verification and grounding, and the contribution is how per-frame answers are turned into reliable 3D evidence:
  distinct-view fusion with negative evidence, optimal-search selection, and uncertainty-triggered magnified
  search. We evaluate on license-free benchmarks (EB-Navigation, RoboTHOR) under the agents' own action spaces.

**Uncertainty, verification and visual prompting for VLMs.**
- Calibrated asking (KnowNo, CoRL 2023) and confidence-driven exploration (Explore-until-Confident, RSS 2024)
  use model uncertainty to decide when to stop or gather information.
- Set-of-Mark and PIVOT elicit grounding through visual marks.
- Our verification probe shows that for a 4B model the bottleneck on appearance-only goals is *recall*: the
  target is among the candidates in under half the frames. It is not the choice between absolute and relative
  verification. S2M therefore spends extra calls on magnified search and multi-view evidence, not on
  re-ranking.
