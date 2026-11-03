"""Decision sources for the two agents, live (calling the local VLM) or replayed from the saved test records.

Every source has: agent, reset(obs), decide(obs) -> (actions, reasoning, perception entry), observe(info, obs),
exhausted(step), occ_map, calls(), and text(T, reasoning, entry, actions, done_count, last_ok) -> (vlm rows, decision rows).
"""
import json
import os
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = 0.8  # S2M commit threshold (frozen configuration)


def s2m_text(T, spec, reasoning, entry, actions, done, last_ok):
    vlm = [(T["phrase"], (spec or {}).get("query", "—"), 2)]
    if entry is None:
        return vlm, [(T["mode"], T["modes"][""], 1)]
    vlm.append((T["visible"], T["pyes"].format(entry["p_vis"]), 1))
    vlm.append((T["where"], T["box"].format([int(v) for v in entry["box"]]) if entry.get("box") else T["nobox"], 1))
    if entry.get("p_zoom") is not None:
        vlm.append((T["crop"], T["pyes"].format(entry["p_zoom"]), 1))
    if entry.get("tiled"):
        vlm.append((T["tiles"], T["tiles_v"].format(entry["tiled"]), 1))
    r = reasoning or {}
    goal = r.get("target") or r.get("goal")
    hyps = entry.get("hyps", [])
    live = [h for h in hyps if not h[3]]
    a = actions[0] if actions else None
    dec = [(T["mode"], T["modes"].get(r.get("mode", ""), r.get("mode", "")), 1),
           (T["goal"], f"({goal[0]:.2f}, {goal[1]:.2f})" if goal else T["none"], 1),
           (T["hyps"], T["hyps_v"].format(len(hyps), sum(h[2] >= COMMIT for h in live), sum(h[3] for h in hyps)), 1)]
    if a is not None:
        dec.append((T["did"] if done else T["next"], T["actions"][a] + (
            "" if last_ok is None else ("  ✓" if last_ok else "  × " + T["rejected"])), 1))
    return vlm, dec


def eb_text(T, reasoning, entry, actions, done, last_ok):
    if reasoning is None:
        return [(T["desc"], "—", 1)], [(T["plan"], "—", 1)]
    r = reasoning
    desc = r.get("visual_state_description", "—")
    invalid = "invalid json" in str(desc)
    acts = [a["action_id"] for a in (r.get("executable_plan") or [])
            if isinstance(a, dict) and isinstance(a.get("action_id"), int) and 0 <= a["action_id"] < 8]
    sep = T["sep"]
    vlm = [(T["desc"], T["invalid"] if invalid else desc, 4),
           (T["reason"], "—" if invalid else r.get("reasoning_and_reflection", "—"), 6)]
    plan = sep.join(T["actions"][a] for a in acts) if acts else T["none"]
    dec = [(T["plan"], plan, 3)]
    if actions:
        dec.append((T["doing"], (T["done_all"] if done >= len(actions) else
                                 f"{done} / {len(actions)} · {T['next']}: {T['actions'][actions[done]]}"), 2))
    if last_ok is not None:
        dec.append((T["feedback"], T["ok"] if last_ok else T["rejected"], 1))
    return vlm, dec


# ---- live sources -------------------------------------------------------------------------------------------

QA = {"zh": {"spec": "把指令改写成要找的目标", "presence": "画面里有「{p}」吗？", "ground": "「{p}」在画面的哪里？",
             "zoom": "放大复核：这个框里是「{p}」吗？", "tile": "切块搜索：这一块里有「{p}」吗？", "yes": "P(是) = {:.2f}",
             "box": "框 [{}]", "none": "没有", "found": "找到", "plan": "计划：{}", "invalid": "输出无法解析"},
      "en": {"spec": "Rewrite the instruction into a target", "presence": "Is there a {p} in the image?",
             "ground": "Where is the {p}?", "zoom": "Crop check: is this box a {p}?", "tile": "Tiled search: a {p} here?",
             "yes": "P(yes) = {:.2f}", "box": "box [{}]", "none": "none", "found": "found", "plan": "plan: {}",
             "invalid": "unparseable output"}}


def instrument_s2m(g, lang, on_call, on_think):
    """Report every VLM call of S2M's grounder (question, answer, latency); nested calls are reported once."""
    L, depth = QA[lang], threading.local()

    def box(b):
        return L["box"].format(", ".join(str(int(v)) for v in b))

    specs = {"target_spec": ("spec", lambda a: L["spec"], lambda r: "「" + r.get("query", "") + "」"),
             "presence": ("presence", lambda a: L["presence"].format(p=a[1]), lambda r: L["yes"].format(r)),
             "ground": ("ground", lambda a: L["ground"].format(p=a[1]),
                        lambda r: box(r[0]) if r and r[0] is not None else L["none"]),
             "zoom_presence": ("zoom", lambda a: L["zoom"].format(p=a[2]), lambda r: L["yes"].format(r)),
             "ground_tile": ("tile", lambda a: L["tile"].format(p=a[1]), lambda r: L["found"] if r is not None else L["none"])}
    for name, (kind, question, answer) in specs.items():
        f = getattr(g, name)

        def wrapped(*a, _f=f, _kind=kind, _q=question, _a=answer, **kw):
            outer = getattr(depth, "n", 0) == 0
            q = _q(a) if outer else None
            if outer:
                on_think(q)
            depth.n = getattr(depth, "n", 0) + 1
            t0 = time.time()
            try:
                r = _f(*a, **kw)
            finally:
                depth.n -= 1
            if outer:
                try:
                    on_call(_kind, q, _a(r), time.time() - t0)
                except Exception:
                    pass
            return r
        setattr(g, name, wrapped)


class S2MLive:
    agent = "s2m"

    def __init__(self, model, skills, lang, on_call, on_think):
        import sys
        sys.path.insert(0, ROOT)
        from embagent.planners import make_planner
        self.p = make_planner("s2m", skills, model_name=model, s2m={})
        instrument_s2m(self.p.g, lang, on_call, on_think)
        self.instruction = ""

    def reset(self, obs):
        self.p.reset(self.instruction)

    def decide(self, obs):
        actions, reasoning = self.p.act(dict(obs, instruction=self.instruction))
        return [int(actions[0])], reasoning, (self.p.log[-1] if self.p.log else None)

    def observe(self, info, obs):
        self.p.observe(info, dict(obs, instruction=self.instruction))

    def exhausted(self, step):
        return False

    @property
    def occ_map(self):
        return self.p.core.map

    def calls(self):
        return self.p.g.calls

    def text(self, T, reasoning, entry, actions, done, last_ok):
        return s2m_text(T, self.p.spec, reasoning, entry, actions, done, last_ok)


class EBLive:
    agent = "eb"

    def __init__(self, model, skills, lang, on_call, on_think):
        import sys
        sys.path.insert(0, ROOT)
        from embagent.planners import make_planner
        self.p = make_planner("eb", skills, model_name=model)
        self.n_skills, self.instruction, self.n_calls = len(skills), "", 0
        model_obj, L = self.p.inner.model, QA[lang]
        orig = model_obj.respond

        def respond(messages, _orig=orig):
            q = {"zh": "根据画面、指令和动作历史生成 JSON 动作计划", "en": "Write a JSON action plan"}[lang]
            on_think(q)
            t0 = time.time()
            out = _orig(messages)
            self.n_calls += 1
            try:
                acts = [skills[a["action_id"]] for a in json.loads(out).get("executable_plan", [])]
                ans = L["plan"].format("; ".join(acts))
            except Exception:
                ans = L["invalid"]
            on_call("plan", q, ans, time.time() - t0)
            return out
        model_obj.respond = respond

    def reset(self, obs):
        self.p.reset(self.instruction)

    def decide(self, obs):
        actions, reasoning = self.p.act(dict(obs, instruction=self.instruction))
        actions = [a if isinstance(a, int) and 0 <= a < self.n_skills else int(np.random.randint(self.n_skills))
                   for a in actions] or [int(np.random.randint(self.n_skills))]
        return actions, reasoning, None

    def observe(self, info, obs):
        self.p.observe(info, obs)

    def exhausted(self, step):
        return False

    occ_map = None

    def calls(self):
        return self.n_calls

    def text(self, T, reasoning, entry, actions, done, last_ok):
        return eb_text(T, reasoning, entry, actions, done, last_ok)

# ---- replayed sources (saved test records) --------------------------------------------------------------------

class S2MRecord:
    agent = "s2m"

    def __init__(self, rec):
        from embagent.planners.geo import GeoCore
        self.rec, self.core, self.prev = rec, GeoCore(), None

    def reset(self, obs):
        self.core.reset(obs["pose"])
        self.core.ensure(dict(obs, step=0))
        self.prev = obs["pose"]

    def decide(self, obs):
        k = obs["step"]
        tr, plans, plog = self.rec["trajectory"], self.rec["plans"], self.rec.get("perception_log", [])
        return [tr[k + 1]["action"]], plans[k]["reasoning"], plog[k] if k < len(plog) else None

    def observe(self, info, obs):
        self.core.record_move(self.prev, info["action_id"], bool(info["last_action_success"]))
        self.core.ensure(obs)
        self.prev = obs["pose"]

    def exhausted(self, step):
        return step >= len(self.rec["trajectory"]) - 1

    @property
    def occ_map(self):
        return self.core.map

    def calls(self):
        return self.rec.get("vlm_calls", 0)

    def text(self, T, reasoning, entry, actions, done, last_ok):
        return s2m_text(T, self.rec.get("spec"), reasoning, entry, actions, done, last_ok)


class EBRecord:
    agent = "eb"
    occ_map = None

    def __init__(self, rec):
        self.rec = rec

    def reset(self, obs):
        pass

    def decide(self, obs):
        k, plans, tr = obs["step"], self.rec["plans"], self.rec["trajectory"]
        i = max(j for j, p in enumerate(plans) if p["step"] <= k)
        end = plans[i + 1]["step"] if i + 1 < len(plans) else len(tr) - 1
        return [t["action"] for t in tr[k + 1:end + 1]], plans[i]["reasoning"], None

    def observe(self, info, obs):
        pass

    def exhausted(self, step):
        return step >= len(self.rec["trajectory"]) - 1

    def calls(self):
        return self.rec.get("planner_calls", 0)

    def text(self, T, reasoning, entry, actions, done, last_ok):
        return eb_text(T, reasoning, entry, actions, done, last_ok)
