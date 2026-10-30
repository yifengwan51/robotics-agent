"""Replay EB-prompt requests that hit max_tokens under a modified decoding setting and check parsing.

  python scripts/replay_truncated.py CAPTURE_DIR [CAPTURE_DIR ...] [--extra JSON] [--bounds JSON] [--out FILE]
                                     [--require-all]

Reads CAPTURE_DIR/truncated_requests_*.jsonl (written by embagent/eb/local_model.py when EB_RAW_CAPTURE_DIR is
set) and resends each request to the running vLLM server (temperature 0, same budget) twice: once unchanged and
once with --extra sampling params and/or the --bounds length limits added to the request's schema. Reports
finish_reason, completion tokens, and whether EB's parser would accept each output. With --require-all, exits 1
unless every modified request parses and none hits max_tokens (used as a gate before a test run).
"""
import argparse
import glob
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embagent.eb.local_model import bounded_schema  # noqa: E402


def eb_parses(text):
    """Same acceptance rule as EBNavigationPlanner.json_to_action: valid JSON with a non-empty action list."""
    try:
        return bool([x["action_id"] for x in json.loads(text)["executable_plan"]])
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dirs", nargs="+")
    ap.add_argument("--extra", default="{}", help='sampling params, e.g. \'{"repetition_penalty": 1.05}\'')
    ap.add_argument("--bounds", default="{}", help='schema bounds, e.g. \'{"max_chars": 1500, "max_items": 20, '
                                                    '"name_chars": 80}\'')
    ap.add_argument("--out", default="", help="JSONL with every replayed output (default: <first dir>/replay.jsonl)")
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    reqs = [json.loads(line) for d in args.capture_dirs
            for f in sorted(glob.glob(os.path.join(d, "truncated_requests_*.jsonl"))) for line in open(f)]
    reqs = reqs[:args.limit] if args.limit else reqs
    extra, bounds = json.loads(args.extra), json.loads(args.bounds)
    out = []
    for i, r in enumerate(reqs):
        row = {"i": i}
        fmt_mod = r["response_format"]
        if bounds:
            fmt_mod = {"type": "json_schema", "json_schema": dict(
                r["response_format"]["json_schema"], schema=bounded_schema(r["response_format"]["json_schema"]["schema"],
                                                                          **bounds))}
        for name, fmt, eb in [("baseline", r["response_format"], {}), ("modified", fmt_mod, extra)]:
            resp = client.chat.completions.create(model=r["model"], messages=r["messages"], response_format=fmt,
                                                  temperature=0, max_tokens=r["max_tokens"], extra_body=eb or None)
            text = resp.choices[0].message.content or ""
            row[name] = {"finish": resp.choices[0].finish_reason, "tokens": resp.usage.completion_tokens,
                         "parses": eb_parses(text), "head": text[:200], "tail": text[-200:]}
        out.append(row)
        print(json.dumps({k: (v if k == "i" else {kk: v[kk] for kk in ("finish", "tokens", "parses")})
                          for k, v in row.items()}), flush=True)
    path = args.out or os.path.join(args.capture_dirs[0], "replay.jsonl")
    with open(path, "w") as fp:
        for row in out:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(out)
    for name in ("baseline", "modified"):
        ok = sum(r[name]["parses"] for r in out)
        cut = sum(r[name]["finish"] == "length" for r in out)
        print(f"{name:9} extra={args.extra if name == 'modified' else '{}'} bounds={args.bounds if name == 'modified' else '{}'}: "
              f"parses {ok}/{n}, still truncated {cut}/{n}")
    if args.require_all:
        good = n > 0 and all(r["modified"]["parses"] and r["modified"]["finish"] != "length" for r in out)
        print("GATE", "PASS" if good else "FAIL")
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
