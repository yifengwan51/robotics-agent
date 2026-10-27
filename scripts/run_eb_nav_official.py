"""Run the official EmbodiedBench EB-Navigation evaluator against a locally served VLM.

The evaluator, environment, planner and prompts are EmbodiedBench's own code; only the model call is
routed to our local OpenAI-compatible server (see embagent/eb/shim.py).

Usage (env embnav_thor, vLLM server already running):
  python scripts/run_eb_nav_official.py --model Qwen2.5-VL-3B-Instruct-AWQ --exp baseline
  python scripts/run_eb_nav_official.py --model ... --eval-sets base --down-sample 0.1   # quick check
Results: runs/eb_nav_official/running/eb_nav/<model>_<exp>/<eval_set>/results/
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embagent.eb import shim  # noqa: E402

shim.install()

import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from embagent.eb.local_model import STATS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name on the local server")
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--eval-sets", nargs="*", default=[], help="default: all five subsets")
    ap.add_argument("--down-sample", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "eb_nav_official"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra config overrides")
    args = ap.parse_args()

    with open(os.path.join(shim.EB_ROOT, "embodiedbench", "configs", "eb-nav.yaml")) as f:
        base = yaml.safe_load(f)
    overrides = dict(model_name=args.model, exp_name=args.exp, eval_sets=args.eval_sets,
                     down_sample_ratio=args.down_sample)
    for kv in args.set:
        key, value = kv.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    cfg = OmegaConf.merge(OmegaConf.create(base), overrides)
    print(OmegaConf.to_yaml(cfg), flush=True)

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)  # EB writes logs relative to the cwd
    from embodiedbench.evaluator.eb_navigation_evaluator import EB_NavigationEvaluator

    evaluator = EB_NavigationEvaluator(cfg)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    print("model call stats:", json.dumps(STATS.snapshot()), flush=True)


if __name__ == "__main__":
    main()
