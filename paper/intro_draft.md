# Introduction (draft)

Multimodal LLM agents can now follow open-ended navigation instructions in simulated homes, but the evidence
comes almost entirely from frontier models behind an API. On EmbodiedBench's EB-Navigation, GPT-4o reaches
57.7% success, while 7-8B open models reach 14-29% and score close to 0% when the target starts behind the
agent [EmbodiedBench]. This gap matters for robotics. A model that runs on the robot's own computer or on an
8 GB consumer GPU has to be small, and every training-free improvement reported so far on this benchmark uses
9B-27B models or API models [EMHO, BrainMem, ETP]. The largest reported gains come from privileged inputs:
simulator-rendered bird's-eye views with the target box drawn in [SeeNav], or reinforcement fine-tuning on
the benchmark's own scenes [SeeNav-SRGPO, ERA].

We first ask *why* small VLM agents fail. A per-frame probe on a held-out split separates two abilities.
Recognition is deciding whether the target is in view and where it is in the image. Metric grounding is
turning that into the right sequence of moves. Small models are imperfect but far from useless at
recognition: a 4B model localizes the named target in [X]% of frames where it is visible, and when it does,
lifting its box with depth puts the target within 0.18 m (median) of its true position. Planning, though,
fails. If the agent is simply told where the target is, a classical geometric planner over the same action
set succeeds in 91.7% of episodes, against [Y]% for the same VLM choosing actions itself. Small VLMs know
*what*; they do not know *where to go*.

This motivates **S2M (Semantic-to-Metric)**, a training-free agent that asks the VLM only questions it can
answer from the current image (is the target visible, which box is it) and leaves every metric decision to
geometry. Three mechanisms make this robust with a small model:
1. **Multi-view evidence fusion.** Detections are lifted into 3D hypotheses that gain evidence from distinct
   viewpoints and lose it when they should be visible but are not detected. The agent commits only when the
   evidence is sufficient, with hysteresis.
2. **Optimal-search selection.** Among competing hypotheses, including same-category look-alikes, the agent
   visits the one with the highest probability-to-cost ratio.
3. **Uncertainty-triggered magnified search.** When nothing is found, extra VLM calls rather than extra
   environment steps go to a tiled, crop-verified search for small and distant targets.

On the official EB-Navigation test set, S2M with a 4-bit Qwen3-VL-4B running on a single RTX 5060 reaches
[Z]% average success, versus [B]% for the same model with the benchmark's prompting and [57.7]% for GPT-4o.
The largest gain is on Long-horizon episodes, [L1] vs [L0]. [Generality across 3 VLM families; robustness to
estimated depth and noisy odometry; RoboTHOR ObjectNav with an explicit Stop, where S2M reaches [R]% SR,
compared with zero-shot methods built on 7B LLMs (47.3) and CLIP (27.5).]

**Contributions.**
1. A diagnosis of small-VLM embodied agents that separates recognition from metric grounding, with per-frame
   probes and oracle ceilings, and the first failure taxonomy for <=8B models on EB-Navigation.
2. S2M, a training-free semantic-to-metric agent with multi-view positive/negative evidence fusion,
   optimal-search hypothesis selection and uncertainty-triggered magnified search.
3. A strict protocol: a held-out development split built with the benchmark's own generator, no privileged
   state, repeated runs with confidence intervals and paired tests, and a full local pipeline for an 8 GB GPU.
