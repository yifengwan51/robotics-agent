"""Pre-registered DEV gate for the simplified S2M (2026-11-10): every module's dev ablation must lower success
(paired McNemar over the 450 dev episodes, Holm over the 4 modules, p < 0.05). Exit 0 = pass, 1 = fail."""
import glob, json, sys
from math import comb
D = "runs/eb_nav/dev/s2m_Qwen3-VL-4B-Instruct-AWQ_"
SUBS = ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"]
def load(run):
    out = {}
    for sfx in ("", "_s1", "_s2"):
        for s in SUBS:
            for f in glob.glob(f"{D}{run}{sfx}/{s}/episode_*.json"):
                out[(sfx, s, f)] = json.load(open(f))["task_success"] >= 1
    return {(k[0], k[1], k[2].split("_")[-1]): v for k, v in out.items()}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
full = load("simp4")
assert len(full) == 450, len(full)
res = []
for m in ("fusion", "scan", "exhaust", "zoom"):
    a = load(f"simp4abl_no_{m}")
    assert len(a) == 450, (m, len(a))
    lose = sum(full[k] and not a[k] for k in full); gain = sum(a[k] and not full[k] for k in full)
    res.append((m, 100 * (sum(a.values()) - sum(full.values())) / 450, lose, gain, mcnemar(lose, gain)))
res.sort(key=lambda r: r[4]); ok = True
for i, (m, d, lose, gain, p) in enumerate(res):
    ph = min(1.0, p * (4 - i)); good = d < 0 and ph < 0.05; ok &= good
    print(f"{m:8s} delta {d:+5.1f} pp  full-only {lose:3d}  abl-only {gain:3d}  p {p:.4f}  Holm {ph:.4f}  {'PASS' if good else 'FAIL'}")
print("full simp4 dev:", round(100 * sum(full.values()) / 450, 1))
sys.exit(0 if ok else 1)
