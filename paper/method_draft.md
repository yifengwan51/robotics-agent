# Method (draft; LaTeX later)

## Problem setting
An agent receives a language instruction $\ell$ that refers to one object and an egocentric RGB-D frame
$(I_t, D_t)$ with odometry pose $x_t = (p_t, \theta_t)$ at each step. It acts with a discrete action set
$\mathcal{A}$ (EB-Navigation: four 0.25 m translations, two 90-degree rotations, two camera tilts) for at most
$T=20$ steps, and succeeds if it ends within 1 m of the target. The VLM $f$ is a small local model ($\le$4B,
4-bit) queried through an OpenAI-compatible API. No simulator state beyond RGB-D, odometry and the
success/failure of the last motion is used.

## Semantic-to-metric decomposition
The VLM-agent baseline asks $f$ for an action sequence. We instead ask it only semantic questions whose
answers are verifiable in the image:
- target spec $s = f(\ell)$: a category name, or an appearance description when $\ell$ only describes looks;
- presence $\pi_t = P_f(\text{yes} \mid I_t, s)$ from the first-token log-probabilities;
- grounding $b_t = f(I_t, s)$, a bounding box or none.

Everything metric is computed. A box is lifted to a 3D point $q_t$ by back-projecting the box centre with the
median depth of its central half (planar pinhole model, known camera height and tilt). The depth stream also
builds a top-down occupancy grid (5 cm cells: floor below 0.1 m, obstacles 0.1-1.9 m). A* runs over
(position, heading) with the benchmark's own actions as edges, each costing one step. The camera only looks
forward, so backward and sideways moves must stay in observed free space ("no blind steps"). Moves the
simulator rejected are stored as forbidden edges.

## Multi-view evidence fusion
Detections are fused into 3D hypotheses $H=\{(\mu_k, w_k)\}$ (merge radius 0.5 m).
- **Positive evidence.** A detection adds $\pi_t$ when presence agrees. When presence disagrees, or the target
  is appearance-only, it adds a crop-verified probability $\pi^{zoom}_t$, from a presence query on a
  2.5x-magnified crop around $b_t$. Repeated looks from the same pose add only a quarter of their weight, so
  confidence grows with view diversity rather than frame count.
- **Negative evidence.** A hypothesis that projects inside the current image, unoccluded according to $D_t$,
  but is not detected loses $\lambda=0.4$.
- **Commitment.** The agent commits when $w_k \ge \tau_c$ and keeps its target while $w_k \ge \tau_k$
  (hysteresis), which stops target flip-flopping.
- **Selection.** Among committed candidates it visits the one maximizing $(1-e^{-w_k})/(c_k+1)$, where $c_k$ is
  the step cost to reach it. This is the optimal-search index for sequential inspection; it reduces to
  nearest-first for equal evidence and strongest-first for equal cost.
- **Exhaustion.** A reached hypothesis that does not end the episode is marked exhausted. On RoboTHOR, a
  failed pre-Stop confirmation does the same.

## Uncertainty-triggered magnified search
If the full frame yields no confident detection and nothing is committed, the frame is split into 3x3
overlapping tiles, each upsampled to 448 px and grounded in parallel. Every tile box must pass crop
verification. The search runs once per (position, heading).

## Search policy
Nothing committed: finish a 360-degree look-around, then inspect the best weak hypothesis, then go to the
vantage point farthest from all visited positions (EB) or the nearest frontier (RoboTHOR).

## Why this helps small VLMs (to be supported by Sec. 3)
Small VLMs lose most of their accuracy on egocentric metric reasoning (left/right, distance, how many steps),
not on recognition. S2M removes metric reasoning from the VLM, converts per-frame semantic answers into 3D
evidence that is checked across views, and spends extra VLM calls (not environment steps) only when evidence
is weak.
