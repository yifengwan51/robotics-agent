# CCF-A Embodied AI on One RTX 5060 + One Quadruped: Research Brief (2026-10-19)

Consolidated from all sweeps, with critic corrections applied. "Est." = estimated.

## 1. Venue plan

Use the CCF 7th edition (released 2026-03-31, erratum 2026-04-09). Cite the official PDF; the ccf.org.cn category pages still show the 6th edition. CCF-A: AAAI, NeurIPS, ACL, CVPR, ICCV, ICML, ICLR (new), ACM MM; journals TPAMI, IJCV, AIJ, JMLR, TIP, TMM (new), SCIS. Only full/regular papers count (no Findings, workshop, short or demo papers).

| Venue | Next deadline | Status | Embodied fit |
|---|---|---|---|
| CVPR 2027 | Paper 2026-11-16 AoE (reg 11-10); decisions 2027-02-25 | Announced | Excellent (VLN/ObjectNav, occupancy, real demos); 4 weeks away |
| ACL 2027 | ARR 2027-01-04 | Announced | Medium; language must be central |
| ICML 2027 | ~late Jan 2027 | Est. | Medium; needs ML-method framing |
| ICCV 2027 (Hong Kong) | ~early Mar 2027 | Est. | Excellent; resubmission after CVPR decisions |
| ACM MM 2027 | ~early Apr 2027 | Est. | Medium; multimodal perception, embodied QA |
| NeurIPS 2027 (+E&D track) | ~early May 2027 | Est. | Very good for learning/efficiency; E&D suits a quadruped benchmark (school credit unverified) |
| AAAI-28 | ~late Jul 2027 | Est. | Very good; legged work accepted at AAAI-26 (ODYSSEY) |
| ICLR 2028 | ~late Sep 2027 | Est. | Very good; one-paper cap without an eligible reciprocal reviewer |
| TPAMI/IJCV/TMM/SCIS | Rolling | n/a | Journal extension; slow |

Passed: ICLR 2027, AAAI-27.

Not CCF-A: ICRA, ECCV, IJCAI (now B), AAMAS and T-RO are B; IROS is C; RSS, CoRL, RA-L, IJRR and Science Robotics are unlisted. TH-CPL (community data) rates RSS/ICRA/ECCV as A, so check your school's 认定目录 first.

Rules: CVPR treats arXiv as non-publication, flags 20% or more overlap as dual submission, bars parallel review from Nov 16, 2026 to Feb 22, 2027, and treats work online after Sep 15, 2026 as contemporaneous. Workshop papers over 4 pages count as prior publication. Supplementary videos must sit inside the upload, anonymized (no logos, faces or robot branding). CVPR allows 8 pages, ICLR 9.

Implication: CVPR 2027 only with results in hand. ICCV 2027 as primary, ACM MM or NeurIPS 2027 as fallback.

## 2. Hardware reality

Verified host: RTX 5060 8 GB (Blackwell, sm_120), driver 580.126 / CUDA 13.0, 32 GB RAM, i7-14700KF (no iGPU), Ubuntu 20.04.6 (glibc 2.31), Python 3.8, Docker with working `--gpus all`. No nvcc, no conda.

VRAM: Xorg runs on the 5060, leaving ~6.27 GiB free (largest single allocation 5.5 GiB). At one check, vLLM, AI2-THOR and other processes held 6.1 of 8 GB at 100% util, so these budgets assume a dedicated GPU. Training from a TTY frees ~1.2 GB.

Measured working on sm_120:
- torch 2.14.1+cu130 (bf16 matmul 38 TFLOPS, SDPA flash, torch.compile); JAX 0.11.2 cuda13.
- Genesis 1.4.4 + rsl-rl: official Go2 PPO, 4096 envs, 100 s, ~2.9 GB peak; 8192 envs OOM.
- MuJoCo Warp 3.16 (2.1M humanoid steps/s); MJX; ManiSkill3/SAPIEN GPU PhysX with Vulkan RGB; bitsandbytes QLoRA.
- Qwen3-VL-2B bf16: 4.07 GiB peak, 82 ms prefill, 70 tok/s. QLoRA 2.76 GiB / 0.26 s/step; LoRA 4.65 GiB.

Hard limits:
- Isaac Gym Preview 4 has no sm_120 kernels and no workaround (NVIDIA), so legged_gym, HIMLoco and SeqWM must be ported.
- pip Isaac Sim ≥5.0, Isaac Lab 2.3 and BEHAVIOR-1K need glibc 2.35, so use containers (pre-built Isaac Sim binaries on 20.04 are hinted, but unofficial). Isaac Sim's minimum is 16 GB VRAM, which rules out camera tasks.
- 7-9B VLMs/VLAs (Uni-NaVid, NaVILA, OpenVLA, Qwen3-VL-8B) don't fit in bf16, int4 is borderline, and full fine-tuning is impossible. Only the 2B model was measured; 4B/8B numbers are estimates.
- No nvcc, so flash-attn, xformers, gsplat-from-source and MASt3R-SLAM extensions need a CUDA-devel container.
- GitHub/HF download at ~0.05 MB/s. Use the Aliyun (~6 MB/s) and ModelScope mirrors.
- Untested: Habitat-sim, MuJoCo Playground (README asks jax[cuda12]), vLLM, Unsloth, Isaac containers, torch cu132.

Recommended stack: leave system Python alone. Use uv-managed CPython 3.12 venvs through the Aliyun mirrors. Venv A: torch 2.14.1+cu130, Genesis, rsl-rl, ManiSkill3, transformers, peft, bitsandbytes. Venv B: JAX, MuJoCo/MJX/Warp. Stay on driver 580 / cu130 (PyTorch dropped cu130 nightlies over a CUDA <13.2.2 miscompilation bug; validate cu132 before moving). Habitat via conda nightly; ROS 2 Humble/Jazzy and Gazebo in Docker.

Models: Qwen3-VL-2B or Qwen3.5-2B as planner (~1 Hz); MiniCPM-V-4.7-1B for a faster loop; YOLOE or Grounding-DINO-tiny, SAM 2.1-small, Depth Anything V2/DA3-small, DINOv2/v3; ViNT/NoMaD baselines; SmolVLA if a VLA is needed. Rent 24-80 GB GPUs for 7-9B baselines and large ablations. The Claude API is unavailable in mainland China/HK. Cloud planner options: Gemini 3.5 Flash-Lite (~$0.80 per 1k image decisions) or Qwen via DashScope (qwen3-vl-plus ~CNY 3 per 1k).

## 3. Robot: the "LTR M1"

Identity is unresolved. No product uses "LTR". Best match (medium confidence): GENISOM/ZsiBot 铜锤 M1 ("中狗 M1 系列" in the SDK). Weaker matches: Lightyear M1 (wheeled quadruped, no SDK found) and Corgi-M1 (DIY). The "LTR" in arXiv 2606.26313 means Load Transfer Ratio.

If it is the GENISOM M1:
- Wheel-foot (16 DOF) or point-foot "F" variants, in base/Pro/Ultra/Air tiers. ~30 kg, 30 kg payload, >4 m/s legged, steps >25 cm, slopes >45°, IP67, ~USD 15k.
- Sensors depend on tier. The 4 fisheye + forward stereo 720° suite is Ultra; UWB, mic/speaker and LiDAR-vision following are Pro. Base-M1 sensors are unknown. A reseller lists dual 96-line LiDAR (unconfirmed).
- Compute: Motion Control Host plus a Navigation Host (mapping, localization, avoidance); platform unpublished. Payload rail 5/12/24/48 V, 480 W, Gigabit Ethernet.
- Public SDK v0.2.2: C++ only, Ubuntu 22.04. High-level only: Move(lateral, forward, yaw) in [-1, 1], held 1 s, sent at 10-20 Hz; ~14 posture modes (Stair, Climb, Sand, ...). Telemetry: IMU up to 100 Hz, odometry 50 Hz, joint pos/vel/torque read-only. MotionData quaternions are wxyz, ImuData xyzw.
- Missing: joint commands, ROS2, and any image, stream, point-cloud or map API. The only camera call is UpdateCameraBitrate.

Consequences:
1. Locomotion is a black box, so RL stays in simulation on a proxy.
2. Every perception, VLN or agent topic is blocked until there is a sensor path: vendor RTSP, Navigation Host ROS2 topics, SSH, or a self-mounted RGB-D camera plus Jetson on the payload rail.
3. Habitat's discrete actions must be mapped to Move(), and the camera height and pitch on the dog measured.

Simulation:
- MATRiX v1.1.2 (MuJoCo 500 Hz + UE5, Zenoh) is the only simulator with an M1-class model. Its zgws/zgwt models are probably the M1 but unconfirmed. Also includes go2/go2w, RGB/depth/fisheye/LiDAR simulation, and the IROSFlatWorld2025, IROSSloppedWorld2025, CrowdWorld and 3DGSWorld maps.
- Locomotion proxies: Go2/Go2-W (unitree_mujoco, Genesis). Navigation benchmarks: Habitat.
- Vendor stacks: genisom_roamerx_open (ROS2 Humble Nav2 + MPPI; hardware docs still TODO) and genisom_vln (UniGoal zero-shot baseline; SmolVLM2, Ollama or DashScope backends).
- Ask GENISOM's 万有引力计划 for a URDF/MJCF, low-level access and Navigation Host deploy rights.

Open questions for the user: nameplate/app photo and variant; how raw camera/LiDAR data reaches the desktop; Navigation Host platform and root access; joint-level access; LiDAR/camera models; site, ground truth (mocap/AprilTag/RTK), number of robots; cloud API budget; target venue.

## 4. State of the art, 2025-2026

### Navigation
Dominant paradigms: training-free modular VLM agents for ObjectNav, and slow-VLM + fast-policy dual systems for real-robot VLN.
- Training-free ObjectNav with richer memory: UniGoal (CVPR 2025), CogNav (ICCV 2025), SG-Nav (NeurIPS 2024), BeliefMapNav (NeurIPS 2025), MSGNav and TrajRAG (CVPR 2026). ApexNav (RA-L 2025) holds the best training-free numbers: HM3Dv1 59.6/33.0, HM3Dv2 76.2/38.0 SR/SPL, with the robot driven from an RTX 4060 Laptop.
- Dual systems: DualVLN/InternVLA-N1 (ICLR 2026; Qwen2.5-VL-7B S2 at ~2 Hz plus a DiT S1 at ~30 Hz; on Go2/G1; R2R-CE 64.3 SR). StreamVLN (ICRA 2026) and NaVILA (RSS 2025) follow the same split.
- Foundation models: NavFoM (ICLR 2026, 4,032 H100-h, real quadruped). CE-Nav (ICLR 2026) is a velocity-level IL-then-RL refiner trained on one RTX 4090.
- Embodiment gap and efficiency: VLN-PE (ICCV 2025) measures a 34% relative SR drop from sim to physical, with low viewpoints hurting quadrupeds. GVNav (ICRA 2025) covers ground-level views. EfficientNav and DynaNav (NeurIPS 2025) target on-device navigation.
- Real-to-sim and world models: EmbodiedSplat (ICCV 2025, phone 3DGS; not the CVPR 2026 paper of the same name) and Navigation World Models (CVPR 2025).
- Caveats: zero-shot VLN-CE often uses 100-episode subsets, and MSGNav fixed an HM3D eval bug. "When Engineering Outruns Intelligence" (arXiv 2507.20021) finds frontier geometry matches LLM planners.

### Agents
Dominant paradigms: VLM agents over explicit memory, calibrated or conformal decisions, and R1-style RL on small VLMs.
- Memory: OpenEQA (CVPR 2024), 3D-Mem (CVPR 2025; GOAT-Bench 69.1 SR with GPT-4o vs 49.6 with LLaVA-7B), MemoryExplorer (CVPR 2026), ReMEmbR (ICRA 2025).
- Calibration: KnowNo (CoRL 2023), Introspective Planning (NeurIPS 2024). Explore-EQA's calibration doesn't transfer across VLMs.
- Small-VLM RL: Embodied-R1 and Cambrian-S (ICLR 2026). EmbodiedBench (ICML 2025) is the common target, and ERA-3B (arXiv) beats GPT-4o on it.
- Compute routing: DIRECT (CoRL 2026; up to 65% lower latency on a Franka) and Sibyl (arXiv; 0.6B SLM, ALFWorld 95.2% at 0.8 cloud calls/episode). Both are manipulation- or text-only.
- Failure and safety: AHA (ICLR 2025), Code-as-Monitor (CVPR 2025) and Talk2Escape (IROS 2026, Go2, asks an oracle). SafeAgentBench shows agents rarely refuse hazardous tasks.

### Perception
Dominant paradigm: feed-forward 3D foundation models replacing per-scene optimization.
- VGGT (CVPR 2025 best paper), MASt3R-SLAM (CVPR 2025), CUT3R (CVPR 2025), VGGT-SLAM (NeurIPS 2025).
- Training-free streaming fixes reach CCF-A: TTT3R (ICLR 2026; ~20 FPS at ~6 GB, ~2x pose gain) and StreamVGGT (ICLR 2026).
- Open-vocabulary maps and graphs: ConceptGraphs (ICRA 2024), HOV-SG (RSS 2024), DualMap (RA-L 2025). Semantic 3DGS is a crowded CVPR 2026 cluster.
- Legged-specific: OneOcc (CVPR 2026; Gait Displacement Compensation, QuadOcc). "Sensor Configuration Matters" (arXiv) finds naive IMU fusion can hurt visual SLAM on ANYmal.
- Promptable real-time perception: YOLOE (ICCV 2025), SAM 3 (ICLR 2026), FoundationStereo (CVPR 2025). CATNAV (arXiv) cuts VLM traversability queries by 85.7%.

### Locomotion
Dominant paradigm: GPU-parallel sim RL. CCF-A acceptances are framed as general learning algorithms, not systems.
- Simulators: Isaac Gym is legacy. MuJoCo Playground, Isaac Lab and Genesis replace it.
- Cheap training: Open-DiffLoco (arXiv; SHAC in MJX, real Go2, <6 GB, 20-60 min on an RTX 5080) and FastTD3.
- Safe and data-efficient sim2real: Real-DRL, SPiDR, DEAL (NeurIPS 2025); Found-adapt (ICLR 2026); proprioceptive distribution matching (arXiv; <5 min of Go2 data).
- World models and theory: WestWorld (ICML 2026 spotlight, Go1 MPPI, checkpoints released), SeqWM (ICLR 2026, real Go2-W), FiRL and ACDA (ICML 2026; random delays, sim only). HIMLoco (ICLR 2024) remains the estimator baseline.
- Generalists such as LocoFormer (CoRL 2025) need a cluster. Wheel-legged RL is still at workshop level.

Not swept: LiDAR-language grounding, audio-visual navigation (RAO-Nav, NeurIPS 2026), outdoor legged navigation (UrbanNav, AAAI-26), embodied visual tracking (e.g., NeurIPS 2026 "Long-term Embodied Visual Tracking with Lightweight VLA Models").

## 5. What gets accepted at CCF-A in embodied AI

Archetypes (share of ~119 CCF-A 2025-26 embodied-nav titles, by title keyword; AAAI and ACM MM not swept):
- Training-free LLM/VLM agents (8.4%): all at CVPR/ICCV/NeurIPS. At ICLR 2026, 25% of rejected vs 5% of accepted nav abstracts claimed zero-shot.
- Trained foundation models/VLAs (11.7%): strongest robot evidence, cluster compute.
- Benchmarks/platforms (10.9%); efficient/on-device (11.7%); world models/imagination (13.4%); 3DGS real-to-sim (5.0%); continual/test-time adaptation (10.9%, mostly sim).
- Analysis papers get in when paired with a reusable platform or a surprising finding plus a fix (VLN-PE; Aux-Think's "inference-time reasoning collapse").
- Legged robots show up mainly as testbeds. OneOcc is the rare legged-titled paper; humanoids appear in titles far more often.

Real-robot norms: a success-rate table against 1-3 baselines. Examples: VLN-PE 14 Go2 episodes, CityWalker 8-14 trials × 3, CE-Nav 40 × 3, UrbanVerse 16 routes × 3, NavFoM 110 cases. Real-world claims lift ICLR 2026 acceptance only modestly (38.6% vs 34.9%); qualitative-only demos are criticized.

Compute norms: low compute is accepted. UniGoal ran local 7B models with no training; ImagineNav used a ResNet-18 plus the GPT-4o-mini API; EvoNav needed 2.4 GB GPU plus an API; CE-Nav used one RTX 4090; VLN-PE baselines took about a day on one 4090.

Reviewer expectations (ICLR 2026 scrape, 799 embodied submissions):
- Baselines are the top complaint (53.9% of reviews).
- Novelty is the clearest accept/reject separator. Rejected zero-shot papers were judged module assembly over SG-Nav/UniGoal/WMNav.
- Training-free claims get audited: API-only comparisons, oracle leakage, unreported cost.
- Several standard suites are expected (R2R-CE, RxR-CE, HM3D, MP3D, HM3D-OVON, GOAT-Bench); single-scene evaluation is penalized.
- Latency and offboard inference get flagged (NavFoM ran on an offboard 4090). Per-module ablations are expected, and the NeurIPS checklist requires error bars and compute.
- Legged datasets need stairs, slopes and hills plus an embodied task (L4Dog was rejected).
- Videos and project pages are de facto standard.
- Base rates (lower bound): 28.4% overall, 35.5% embodied, 44.2% navigation.

## 6. Opportunity seeds

Prerequisites for all seeds: a sensor path from the M1, a dedicated GPU, and a novelty re-check against NeurIPS 2026 accepts and Jun-Oct 2026 arXiv. CVPR treats work posted after 2026-09-15 as contemporaneous.

1. **Gait/viewpoint-robust zero-shot navigation.** Idea: model low camera height, pitch oscillation and blur inside a training-free ObjectNav/VLN agent; reproduce the drop in Habitat, fix it, validate on the M1. Why now: zero-shot CMA gets only 4.73 SR on a quadruped in VLN-PE, and OneOcc shows gait jitter matters. Prior: VLN-PE (ICCV 2025), GVNav (ICRA 2025). Risk: the diagnosis exists, so the fix has to carry the paper.
2. **Calibrated goal verification and stopping.** Idea: an uncertainty-gated (conformal) stop/verify module for open-vocabulary goals on top of ApexNav or VLFM, tested on full HM3D/MP3D/OVON splits plus real trials. Why now: AnyGoal finds failures shifting from exploration to verification, and NaVid spins near the goal in 70% of VLN-PE episodes. Prior: ApexNav (RA-L 2025), Explore-EQA (RSS 2024). Risk: guarantees may break under sim-to-real shift.
3. **Risk-controlled local/cloud VLM cascade.** Idea: a 4-bit local VLM with per-step conformal escalation to an API VLM for EQA and object search; report accuracy, calls, latency and cost. Why now: WMNav falls from 58.1 SR with Gemini 1.5 Pro to 29.7 with a 3B model. Prior: Sibyl (arXiv, text-only), DIRECT (CoRL 2026, manipulation), KnowNo. Risk: RACER may own the theory; needs DashScope or Gemini access.
4. **Latency- and delay-aware asynchronous planning.** Idea: learn when to call the VLM and correct stale plans while the dog moves under 1 s held Move() commands and variable Wi-Fi delay. Why now: DualVLN's S2 takes 0.7-1.1 s, and ACDA handles random delays in sim only. Prior: SysNav (arXiv), ACDA (ICML 2026). Risk: may read as engineering unless it generalizes.
5. **Sub-8 GB navigation VLA.** Idea: combine 4-bit weights, training-free token pruning and KV reuse on a released VLN model; plot accuracy vs latency on R2R-CE/RxR-CE, the 5060 and the dog. Why now: NaVILA needs 8.6 GB even at W4A16, and reviewers flag offboard 4090s. Prior: EfficientNav (NeurIPS 2025), token pruning on Go2 (IROS 2026). Risk: crowded; pruning doesn't shrink weights.
6. **Cloud-to-local distillation with auxiliary CoT.** Idea: log a cloud teacher's frontier, stop and CoT decisions on HM3D train scenes, then QLoRA a 2-3B VLM with CoT as training-only supervision. Why now: Aux-Think shows test-time CoT hurts, and 2B QLoRA costs 2.76 GiB here. Prior: Aux-Think (NeurIPS 2025). Risk: teacher API cost; gains may vanish on the real robot.
7. **Legged real-to-sim evaluation.** Idea: phone-captured 3DGS replicas of your building for semantic-goal navigation; measure quadruped sim-real correlation over matched episodes. Why now: EmbodiedSplat's 0.87-0.97 correlation is ImageNav on a wheeled Stretch, and VLN-PE has only 14 real episodes. Prior: EmbodiedSplat (ICCV 2025), ReaDy-Go (arXiv). Risk: leg dynamics and moving people may weaken correlation.
8. **Proprioception-aware streaming reconstruction.** Idea: gate CUT3R/TTT3R state updates with IMU and leg odometry, training-free; evaluate on TUM, GrandTour and M1 logs. Why now: naive IMU fusion hurts on legs, and TTT3R fits in ~6 GB. Prior: TTT3R (ICLR 2026). Risk: crowded follow-ups; camera-IMU sync is undocumented in the SDK.
9. **Dog-view long-term memory under change.** Idea: multi-day, multi-floor, indoor-outdoor recordings with change-aware QA, plus a memory agent with change detection and calibrated abstention. Why now: 3D-Mem lacks multi-floor support, ReMEmbR assumes a static world, and FindingDory/EMem-Bench are simulated. Prior: 3D-Mem (CVPR 2025), MemoryExplorer (CVPR 2026). Risk: overlap with GangnamLoop, GuideDogQA and Follow-Bench 2.0; heavy collection.
10. **Capability-conditioned closed loop.** Idea: VLM traversability reasoning picks routes and M1 posture modes (Stair, Climb, Sand) and executes them. Why now: CapNav shows VLM navigation degrades as mobility constraints tighten but tests no robot. Prior: CapNav (CVPR 2026), CATNAV (arXiv). Risk: posture modes are vendor black boxes; stair trials are risky.

11. **Navigation-level failure monitoring.** Idea: VLM-written runtime checks (stuck, looping, wrong room, target lost, unsafe terrain) with escalation and replanning; build a failure taxonomy from real M1 runs plus Habitat. Why now: AHA and Code-as-Monitor are manipulation-only, and Talk2Escape only detects looping and asks an oracle. Prior: Code-as-Monitor (CVPR 2025), Talk2Escape (IROS 2026). Risk: natural failures are slow to collect and label.
12. **Proprioceptive failure prediction as a safety layer.** Idea: fine-tune a trajectory world model on M1 joint and IMU telemetry to predict slips and falls and gate Move(). Why now: WestWorld checkpoints are released, and the SDK exposes read-only joint telemetry. Prior: WestWorld (ICML 2026), Real-DRL (NeurIPS 2025). Risk: few real falls; the vendor controller may mask precursors.
13. **Velocity-level refiner with minute-scale calibration.** Idea: train a local navigation refiner in MATRiX/Genesis against Move(), calibrated by distribution matching on about 5 min of real odometry. Why now: CE-Nav trained on one 4090, and distribution matching cut Go2 drift with under 5 min of data. Prior: CE-Nav (ICLR 2026). Risk: needs depth or LiDAR input; the zgws/zgwt model is unconfirmed.
14. **Quadruped person-following and social navigation.** Idea: a lightweight VLM/VLA follower with social-norm reasoning, tested in MATRiX CrowdWorld and on campus. Why now: VLMs trail rule-based methods on social understanding (SocialNav-SUB), and Pro-tier UWB could supply ground truth. Prior: EVT with lightweight VLA (NeurIPS 2026). Risk: concurrent work; filming consent.
15. **Open-model, cost-reported evaluation protocol.** Idea: re-run 3-4 zero-shot agents with small local VLMs on full splits plus matched real-dog trials; report accuracy, latency, tokens and cost; test when language beats geometry. Why now: subset evaluation, the HM3D eval bug and API-only critiques. Prior: arXiv 2507.20021, VLN-PE (ICCV 2025). Risk: needs a surprising finding plus a fix to clear CCF-A.

