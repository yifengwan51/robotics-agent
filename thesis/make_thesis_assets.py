"""Facts and tables used only by the thesis, computed from the saved test records (no number is typed by hand).

  python thesis/make_thesis_assets.py
    -> chapters/generated/thesis_numbers.tex   timing and environment facts
    -> chapters/generated/tab_trace.tex        evidence of every hypothesis in one Long-horizon episode of S2M
    -> chapters/generated/tab_eb_excerpt.tex   what the EB prompt writes about the target vs. what the simulator shows
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "runs", "eb_nav", "test")
OUT = os.path.join(ROOT, "thesis", "chapters", "generated")
S2M, EB, MOD = ("s2m_Qwen3-VL-4B-Instruct-AWQ_simp4_r1", "eb_Qwen3-VL-4B-Instruct-FP8_eb_official_r1",
                "s2m_Qwen3-VL-4B-Instruct-AWQ_simp4_modular_r1")
M = {}


def recs(run):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(TEST, run, "*", "episode_*.json")))]


def tex(s):
    return s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


# timing of S2M on one 8 GB GPU that also renders the simulator
r = recs(S2M)
el, ns = [x["elapsed_s"] for x in r], [x["num_steps"] for x in r]
M["TimeStepS"] = f"{sum(el) / sum(ns):.2f}"
M["TimeEpS"] = f"{st.mean(el):.1f}"
M["StepsMeanS"] = f"{st.mean(ns):.1f}"

# evidence trace: Long-horizon test episode 42 (the episode of the thumbnails in the architecture figure)
ep = json.load(open(os.path.join(TEST, S2M, "long_horizon", "episode_42.json")))
target = ep["target"]
ACT = ["forward", "backward", "right", "left", "turn right", "turn left", "tilt up", "tilt down"]
near = lambda x, z: ((x - target["x"]) ** 2 + (z - target["z"]) ** 2) ** 0.5 < 0.5
with open(os.path.join(OUT, "tab_trace.tex"), "w") as f:
    f.write("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}rlcccrl@{}}\n\\toprule\n"
            "Step & Observation & $\\pi_t$ & $\\pi^z_t$ & $a_t$ & $w$ (target) & $w$ (look-alikes) & Action\\\\\n"
            .replace("rlcccrl", "rlcccrll") + "\\midrule\n")
    for e in ep["perception_log"]:
        step = e["step"]
        if e.get("box") is None:
            obs = "no box"
        elif not e["dets"]:
            obs = "box, rejected"
        else:
            obs = "box on target" if near(*e["dets"][0][:2]) else "box on look-alike"
        pz = "not asked" if e.get("p_zoom") is None else f"{e['p_zoom']:.2f}"
        a_t = f"{e['dets'][0][2]:.2f}" if e["dets"] else "0"
        wt = next((h[2] for h in e["hyps"] if near(h[0], h[1])), None)
        wt = "none" if wt is None else (f"\\textbf{{{wt:.2f}}}" if wt >= 0.8 else f"{wt:.2f}")
        la = [h[2] for h in e["hyps"] if not near(h[0], h[1])]
        la = "none" if not la else ", ".join(f"{w:.2f}" for w in la)
        nxt = ep["trajectory"][step + 1]["action"] if step + 1 < len(ep["trajectory"]) else None
        act = "reached" if nxt is None else ACT[nxt]
        f.write(f"{step} & {obs} & {e['p_vis']:.2f} & {pz} & {a_t} & {wt} & {la} & {act}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
M["TraceSteps"] = str(ep["num_steps"])
M["TraceTarget"] = tex(ep["instruction"].split("navigate to the ")[1].split(" in the room")[0])
M["TraceFinalW"] = f"{next(h[2] for h in ep['perception_log'][-1]['hyps'] if near(h[0], h[1])):.2f}"

# EB prompt excerpt: Long-horizon test episode 14, where the model reports a target that is not in view
eb = json.load(open(os.path.join(TEST, EB, "long_horizon", "episode_14.json")))
skills = ["forward", "backward", "right", "left", "turn right", "turn left", "tilt up", "tilt down"]
lines = []
starts = [p["step"] for p in eb["plans"]] + [eb["num_steps"]]
for i, p in enumerate(eb["plans"][:4]):
    t = eb["trajectory"][p["step"]]
    claim = p["reasoning"].get("visual_state_description", "")
    first = claim.split(". ")[0].rstrip(".") + "."
    acts = ", ".join(skills[a] for a in p["actions"])
    span = eb["trajectory"][p["step"] + 1:starts[i + 1] + 1]  # actions executed for this plan (until one fails)
    ok, bad = sum(1 for x in span if x["ok"]), sum(1 for x in span if not x["ok"])
    lines.append((p["step"], first, "yes" if t["vis"] else "no", acts, f"{ok} / {bad}"))
with open(os.path.join(OUT, "tab_eb_excerpt.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}rp{0.42\\linewidth}cp{0.17\\linewidth}c@{}}\n\\toprule\n"
            "Step & Model's description of the view & In view & Plan & Acc./rej.\\\\\n\\midrule\n")
    for step, first, vis, acts, ok in lines:
        f.write(f"{step} & \\textit{{``{tex(first)}''}} & {vis} & {acts} & {ok}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
M["ExcerptRejected"] = str(sum(1 for t in eb["trajectory"][1:] if not t["ok"]))
M["ExcerptSteps"] = str(eb["num_steps"])
M["ExcerptSeen"] = "never" if not any(t["vis"] for t in eb["trajectory"]) else "at some step"
s14 = json.load(open(os.path.join(TEST, S2M, "long_horizon", "episode_14.json")))
M["ExcerptSSteps"] = str(s14["num_steps"])
M["ExcerptSOk"] = "succeeds" if s14["task_success"] >= 1 else "fails"

# where the remaining failures of S2M (Qwen3-VL-4B, four test runs) fall
import sys
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_nav_failures import classify  # noqa: E402
lost = {"all": 0, "visual_appearance": 0}
for i in (1, 2, 3):
    for x in recs(f"s2m_Qwen3-VL-4B-Instruct-AWQ_simp4_r{i}"):
        if classify(x) == "lost_after_seen":
            lost["all"] += 1
            lost["visual_appearance"] += x["eval_set"] == "visual_appearance"
M["LostSVisual"] = f"{lost['visual_appearance'] / 3:.1f}"
M["LostSAll"] = f"{lost['all'] / 3:.1f}"

with open(os.path.join(OUT, "thesis_numbers.tex"), "w") as f:
    for k, v in M.items():
        f.write(f"\\newcommand{{\\t{k}}}{{{v}}}\n")
print(M)
