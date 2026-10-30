"""Output-validity check for an EB-prompt run: JSON decode failures in the episode records and responses that hit
max_tokens in the raw capture.

  python scripts/check_eb_outputs.py RUN_DIR CAPTURE_DIR [--gate]

With --gate, exits 1 unless the run has no JSON decode failure and no response finished with reason "length".
"""
import argparse
import collections
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("capture_dir")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    eps = [json.load(open(f)) for f in glob.glob(os.path.join(args.run_dir, "*", "episode_*.json"))]
    rows = [json.loads(line) for f in glob.glob(os.path.join(args.capture_dir, "responses_*.jsonl")) for line in open(f)]
    errs = sum(e.get("json_errors", 0) for e in eps)
    calls = sum(e.get("model_calls", {}).get("calls", 0) for e in eps)
    fin = collections.Counter(r["finish_reason"] for r in rows)
    sr = 100 * sum(e.get("task_success", 0) >= 1 for e in eps) / max(1, len(eps))
    toks = sorted(r["completion_tokens"] or 0 for r in rows)
    print(f"{args.run_dir}: {len(eps)} episodes, SR {sr:.1f}%, {calls} planner calls, JSON decode failures {errs}, "
          f"episodes affected {sum(e.get('json_errors', 0) > 0 for e in eps)}, runner errors "
          f"{sum('error' in e for e in eps)}")
    print(f"  capture: {len(rows)} responses, finish {dict(fin)}, completion tokens p50 "
          f"{toks[len(toks) // 2] if toks else '-'} max {toks[-1] if toks else '-'}")
    bounds = json.load(open(os.path.join(args.run_dir, "args.json"))).get("vlm_schema_bounds") or {}
    over = 0
    if bounds:  # every parsed output must respect the schema bounds the run was recorded with
        for r in rows:
            try:
                o = json.loads(r["raw"])
            except Exception:
                continue
            over += any(isinstance(v, str) and len(v) > bounds["max_chars"] for v in o.values())
            over += len(o.get("executable_plan", [])) > bounds["max_items"]
        print(f"  schema bounds {bounds}: outputs violating them {over}")
    if args.gate:
        good = len(eps) > 0 and errs == 0 and fin.get("length", 0) == 0 and not any("error" in e for e in eps) \
            and over == 0
        print("GATE", "PASS" if good else "FAIL")
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
