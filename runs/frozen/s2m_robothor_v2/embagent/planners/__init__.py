"""Planners for EB-Navigation-style episodes.

Interface (used by embagent.eb.nav_runner):
  reset(instruction)                 at episode start
  act(obs) -> (action_ids, reasoning_dict)
  observe(info, obs)                 after every executed env step
obs = {"rgb": HxWx3 uint8, "depth": HxW float32 metres, "pose": {"x","z","yaw","horizon"}, "instruction": str, "step": int}
"""
import os
import tempfile

import numpy as np
from PIL import Image


class Planner:
    name = "base"

    def reset(self, instruction):
        pass

    def act(self, obs):
        raise NotImplementedError

    def observe(self, info, obs):
        pass


class RandomPlanner(Planner):
    """Plumbing check without a model."""
    name = "random"

    def __init__(self, n_actions, seed=0):
        self.n, self.rng = n_actions, np.random.default_rng(seed)

    def act(self, obs):
        return [int(self.rng.integers(self.n))], {"reasoning_and_reflection": "random"}


class EBPromptPlanner(Planner):
    """EmbodiedBench's own EBNavigationPlanner (same prompts, examples, parsing, random fallback on bad JSON).

    Only difference from the official evaluator: the frame is written to a per-process temp PNG instead of
    the run directory. Requires embagent.eb.shim.install() before construction.
    """
    name = "eb"

    def __init__(self, model_name, actions, n_shots=3, chat_history=False, truncate=False):
        from embodiedbench.evaluator.config.eb_navigation_example import examples
        from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
        from embodiedbench.planner.nav_planner import EBNavigationPlanner

        self.inner = EBNavigationPlanner(
            model_name=model_name, model_type="remote", actions=actions,
            system_prompt=eb_navigation_system_prompt, examples=examples, n_shot=n_shots, obs_key="head_rgb",
            chat_history=chat_history, language_only=False, multiview=False, multistep=False,
            visual_icl=False, truncate=truncate)
        self._img = os.path.join(tempfile.gettempdir(), f"embagent_frame_{os.getpid()}.png")
        self._instruction = ""

    def reset(self, instruction):
        self.inner.reset()
        self._instruction = instruction

    def act(self, obs):
        import json

        Image.fromarray(obs["rgb"]).save(self._img)
        action, out = self.inner.act(self._img, self._instruction)
        try:
            reasoning = json.loads(out)
        except Exception:
            reasoning = {"raw": out}
        if not isinstance(action, list):
            action = [action]
        return [int(a) for a in action], reasoning

    def observe(self, info, obs):
        self.inner.update_info(info)

    @property
    def stats(self):
        return {"planner_steps": self.inner.planner_steps, "json_errors": self.inner.output_json_error}


def make_planner(name, actions, **kw):
    if name == "eb":
        return EBPromptPlanner(kw["model_name"], actions, n_shots=kw.get("n_shots", 3),
                               chat_history=kw.get("chat_history", False), truncate=kw.get("truncate", False))
    if name == "random":
        return RandomPlanner(len(actions), seed=kw.get("seed", 0))
    if name == "oracle_geo":
        from embagent.planners.geo import OracleGeoPlanner
        return OracleGeoPlanner(face_target=kw.get("face_target", False))
    if name == "s2m":
        from embagent.planners.s2m import S2MPlanner
        return S2MPlanner(kw["model_name"], **kw.get("s2m", {}))
    raise ValueError(f"unknown planner {name!r}")
