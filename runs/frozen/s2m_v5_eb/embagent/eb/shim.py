"""Import EmbodiedBench modules without its heavy optional deps (lmdeploy, torch, anthropic, google-generativeai).

install() must run before any `embodiedbench.*` import. It:
- puts third_party/EmbodiedBench on sys.path,
- replaces `embodiedbench.main` (hydra + a cwd symlink side effect) with a logger-only module,
- replaces `embodiedbench.planner.remote_model` / `custom_model` with our local OpenAI-compatible client,
- stubs `google.generativeai`, which planner_utils imports but EB-Navigation never calls.
The benchmark code itself (envs, evaluator, planner, prompts) is used unmodified.
"""
import logging
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EB_ROOT = os.path.join(ROOT, "third_party", "EmbodiedBench")

_installed = False


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install():
    global _installed
    if _installed:
        return
    if EB_ROOT not in sys.path:
        sys.path.insert(0, EB_ROOT)

    logger = logging.getLogger("EB_logger")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    _module("embodiedbench.main", logger=logger)

    if "google.generativeai" not in sys.modules:
        google = sys.modules.get("google") or _module("google")
        genai = _module("google.generativeai")
        google.generativeai = genai

    from embagent.eb.local_model import LocalOpenAIModel

    class RemoteModel(LocalOpenAIModel):
        """Drop-in for embodiedbench.planner.remote_model.RemoteModel (same constructor signature)."""

        def __init__(self, model_name, model_type="remote", language_only=False, tp=1, task_type=None):
            super().__init__(model_name, language_only=language_only, task_type=task_type)

    class CustomModel:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("custom model_type is not used in this project")

    _module("embodiedbench.planner.remote_model", RemoteModel=RemoteModel)
    _module("embodiedbench.planner.custom_model", CustomModel=CustomModel)
    _installed = True
