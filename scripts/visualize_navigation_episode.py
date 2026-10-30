#!/usr/bin/env python3
"""Create a self-contained HTML replay for one recorded EB-Navigation episode.

Example:
  python scripts/visualize_navigation_episode.py \
    --s2m-run runs/eb_nav/test/s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1 \
    --baseline-run runs/eb_nav/test/eb_Qwen3-VL-4B-Instruct-AWQ_eb_official_r1 \
    --subset long_horizon --episode 42 --output outputs/navigation_episode_42.html

The script reads existing episode JSON only. It never initializes a simulator or model.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


def read_record(path: Path) -> dict[str, Any]:
    """Read a normal JSON result; tolerate concatenated JSON objects in legacy logs."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value is not an object")
        return value
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        pos = 0
        objects: list[dict[str, Any]] = []
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            value, pos = decoder.raw_decode(text, pos)
            if isinstance(value, dict):
                objects.append(value)
        if not objects:
            raise
        # Legacy action logs have one object per step. Keep them replayable even
        # when a full trajectory summary was not saved.
        return {
            "instruction": objects[0].get("instruction", "unknown instruction"),
            "trajectory": [
                {
                    "action": item.get("action_id"),
                    "ok": item.get("last_action_success"),
                    "dist": item.get("distance"),
                }
                for item in objects
            ],
            "legacy_steps": objects,
        }


def find_episode(run_dir: Path, subset: str, episode: int) -> Path:
    """Resolve one formal result, excluding duplicated action-level `_eb_logs`."""
    path = run_dir / subset / f"episode_{episode}.json"
    if not path.is_file():
        raise FileNotFoundError(f"episode_{episode}.json not found at {path}")
    return path


def compact_plan(record: dict[str, Any], step: int) -> dict[str, Any] | None:
    plans = record.get("plans") or []
    selected = None
    for plan in plans:
        if int(plan.get("step", -1)) <= step:
            selected = plan
        else:
            break
    if not selected:
        return None
    reasoning = selected.get("reasoning") or {}
    return {
        "step": selected.get("step"),
        "actions": selected.get("actions", []),
        "mode": reasoning.get("mode"),
        "score": reasoning.get("score"),
        "text": reasoning.get("language_plan") or reasoning.get("reasoning_and_reflection") or "",
    }


def prepare(record: dict[str, Any], label: str) -> dict[str, Any]:
    trajectory = record.get("trajectory") or []
    perception_by_step = {item.get("step"): item for item in record.get("perception_log") or []}
    frames = []
    for step, state in enumerate(trajectory):
        item = dict(state)
        item["step"] = step
        item["perception"] = perception_by_step.get(step)
        item["plan"] = compact_plan(record, step)
        frames.append(item)
    return {
        "label": label,
        "instruction": record.get("instruction", "unknown instruction"),
        "scene": record.get("scene", "not recorded"),
        "success": record.get("task_success"),
        "final_distance": record.get("final_distance"),
        "planner_calls": record.get("planner_calls", record.get("vlm_calls")),
        "model_calls": record.get("model_calls", {}).get("calls", record.get("vlm_calls")),
        "target": record.get("target"),
        "frames": frames,
        "source_has_coordinates": all("x" in frame and "z" in frame for frame in frames),
    }


def build_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = escape(f"Navigation replay — episode {payload['episode']}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f6f7fb; --card:#fff; --text:#172033; --muted:#607087; --line:#d7ddea; --base:#667085; --s2m:#1769aa; --target:#d14b43; --ok:#258a58; --bad:#c23b3b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151922; --card:#202633; --text:#edf2fa; --muted:#b5c0d1; --line:#3a4353; --base:#b7c0cf; --s2m:#65b6ef; --target:#ff827b; --ok:#61cf92; --bad:#fa7d78; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px; }} h1 {{ font-size:22px; margin:0 0 5px }} h2 {{ font-size:16px; margin:0 0 8px }} .muted {{ color:var(--muted) }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:16px }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }}
.meta {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:12px }} .metric {{ border-left:3px solid var(--line); padding-left:8px }} .metric b {{ display:block; font-size:16px }}
canvas {{ width:100%; aspect-ratio:1.35; display:block; border:1px solid var(--line); border-radius:6px; background:color-mix(in srgb,var(--card) 88%,var(--bg)); }}
.controls {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:16px 0 8px }} input[type=range] {{ flex:1; min-width:220px }} button {{ border:1px solid var(--line); background:var(--card); color:var(--text); border-radius:6px; padding:6px 11px; cursor:pointer; }}
.detail {{ white-space:pre-wrap; min-height:72px; max-height:180px; overflow:auto; color:var(--muted); font-size:13px }} .legend {{ display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:12px }} .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px }}
@media (max-width:760px) {{ main {{ padding:14px }} .grid {{ grid-template-columns:1fr }} .meta {{ grid-template-columns:repeat(2,minmax(0,1fr)) }} }}
</style></head><body><main>
<h1>Recorded navigation replay: episode {payload['episode']}</h1><div id="instruction" class="muted"></div>
<div class="controls"><button id="play" type="button">Play</button><button id="prev" type="button">‹ Step</button><button id="next" type="button">Step ›</button><input id="slider" type="range" min="0" value="0"><strong id="stepLabel"></strong></div>
<div class="legend"><span><i class="dot" style="background:var(--s2m)"></i>S2M trajectory / hypotheses</span><span><i class="dot" style="background:var(--base)"></i>EB-prompt trajectory</span><span><i class="dot" style="background:var(--target)"></i>true target (analysis only)</span><span><i class="dot" style="background:var(--ok)"></i>target visible</span></div>
<section class="grid"><article class="card"><h2>S2M</h2><canvas id="s2mCanvas"></canvas><div id="s2mMeta" class="meta"></div><h2 style="margin-top:14px">Current semantic evidence</h2><div id="s2mDetail" class="detail"></div></article><article class="card"><h2>EB prompt baseline</h2><canvas id="baseCanvas"></canvas><div id="baseMeta" class="meta"></div><h2 style="margin-top:14px">Current planner output</h2><div id="baseDetail" class="detail"></div></article></section>
</main><script id="payload" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('payload').textContent), S=D.s2m, B=D.baseline;
const maxStep=Math.max(S.frames.length,B.frames.length)-1, slider=document.getElementById('slider'); slider.max=Math.max(0,maxStep);
document.getElementById('instruction').textContent=S.instruction||B.instruction;
const colors={{s2m:getComputedStyle(document.documentElement).getPropertyValue('--s2m').trim(),base:getComputedStyle(document.documentElement).getPropertyValue('--base').trim(),target:getComputedStyle(document.documentElement).getPropertyValue('--target').trim(),ok:getComputedStyle(document.documentElement).getPropertyValue('--ok').trim(),text:getComputedStyle(document.documentElement).getPropertyValue('--text').trim(),line:getComputedStyle(document.documentElement).getPropertyValue('--line').trim()}};
function coords(run){{return run.frames.filter(f=>Number.isFinite(f.x)&&Number.isFinite(f.z));}}
const all=[...coords(S),...coords(B),...(S.target?[{{x:S.target.x,z:S.target.z}}]:[])];
const xs=all.map(p=>p.x), zs=all.map(p=>p.z); let loX=Math.min(...xs),hiX=Math.max(...xs),loZ=Math.min(...zs),hiZ=Math.max(...zs); const pad=Math.max(.35,(hiX-loX)*.14,(hiZ-loZ)*.14); loX-=pad;hiX+=pad;loZ-=pad;hiZ+=pad;
function frame(run,i){{return run.frames[Math.min(i,run.frames.length-1)]||null;}}
function metric(run){{return `<div class="metric"><span class="muted">Outcome</span><b>${{run.success===1||run.success===1.0?'Success':run.success===0||run.success===0.0?'Failure':'Not saved'}}</b></div><div class="metric"><span class="muted">Final distance</span><b>${{Number.isFinite(run.final_distance)?run.final_distance.toFixed(2)+' m':'--'}}</b></div><div class="metric"><span class="muted">Planner calls</span><b>${{run.planner_calls??'--'}}</b></div><div class="metric"><span class="muted">Frames</span><b>${{run.frames.length}}</b></div>`;}}
document.getElementById('s2mMeta').innerHTML=metric(S); document.getElementById('baseMeta').innerHTML=metric(B);
function draw(canvas,run,i,color){{const r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(r.width*dpr));canvas.height=Math.max(1,Math.round(r.height*dpr));const c=canvas.getContext('2d');c.scale(dpr,dpr);const w=r.width,h=r.height,m=30;const scale=Math.min((w-2*m)/(hiX-loX||1),(h-2*m)/(hiZ-loZ||1));const p=q=>[m+(q.x-loX)*scale,h-m-(q.z-loZ)*scale];c.strokeStyle=colors.line;c.lineWidth=1;c.strokeRect(m,m,w-2*m,h-2*m);c.fillStyle=colors.text;c.font='12px system-ui';c.fillText('top-down x/z (m)',m,18);const pts=coords(run).slice(0,Math.min(i+1,run.frames.length));if(pts.length){{c.beginPath();pts.forEach((q,n)=>{{const [x,y]=p(q);n?c.lineTo(x,y):c.moveTo(x,y)}});c.strokeStyle=color;c.lineWidth=3;c.stroke();pts.forEach((q,n)=>{{const [x,y]=p(q);c.fillStyle=color;c.globalAlpha=n===pts.length-1?1:.42;c.beginPath();c.arc(x,y,n===pts.length-1?6:3,0,Math.PI*2);c.fill()}});c.globalAlpha=1;const q=pts[pts.length-1];const [x,y]=p(q);const yaw=(q.yaw||0)*Math.PI/180;c.strokeStyle=color;c.lineWidth=2;c.beginPath();c.moveTo(x,y);c.lineTo(x+16*Math.sin(yaw),y-16*Math.cos(yaw));c.stroke();}}
if(S.target){{const [x,y]=p({{x:S.target.x,z:S.target.z}});c.strokeStyle=colors.target;c.lineWidth=3;c.beginPath();c.moveTo(x-7,y-7);c.lineTo(x+7,y+7);c.moveTo(x+7,y-7);c.lineTo(x-7,y+7);c.stroke();}}
if(run===S){{const f=frame(S,i), hyps=f?.perception?.hyps||[];hyps.forEach(h=>{{const [x,y]=p({{x:h[0],z:h[1]}});c.fillStyle=colors.s2m;c.globalAlpha=.35;c.beginPath();c.arc(x,y,Math.max(4,Math.min(13,Math.abs(h[2])*1.5)),0,Math.PI*2);c.fill()}});c.globalAlpha=1;}}
}}
function detail(run,i,isS2M){{const f=frame(run,i);if(!f)return 'No step record.';const common=`action: ${{f.action??'start'}}\nexecuted: ${{f.ok===undefined?'not recorded':f.ok}}\ndistance: ${{Number.isFinite(f.dist)?f.dist.toFixed(3)+' m':'not recorded'}}\ntarget visible: ${{f.vis===undefined?'not recorded':f.vis}}`;if(isS2M){{const p=f.perception||{{}};return common+`\nmode: ${{f.plan?.mode??'not recorded'}}\nVLM visibility probability: ${{p.p_vis??'not recorded'}}\nzoom probability: ${{p.p_zoom??'not used'}}\ntiled search: ${{p.tiled??'not recorded'}}\nhypotheses: ${{JSON.stringify(p.hyps??[])}}`;}}return common+`\nplanned actions: ${{JSON.stringify(f.plan?.actions??[])}}\nplan: ${{f.plan?.text??'not recorded'}}`;}}
let timer=null;function render(){{const i=+slider.value;document.getElementById('stepLabel').textContent=`Step ${{i}} / ${{maxStep}}`;draw(document.getElementById('s2mCanvas'),S,i,colors.s2m);draw(document.getElementById('baseCanvas'),B,i,colors.base);document.getElementById('s2mDetail').textContent=detail(S,i,true);document.getElementById('baseDetail').textContent=detail(B,i,false);}}
document.getElementById('prev').onclick=()=>{{slider.value=Math.max(0,+slider.value-1);render()}};document.getElementById('next').onclick=()=>{{slider.value=Math.min(maxStep,+slider.value+1);render()}};slider.oninput=render;document.getElementById('play').onclick=e=>{{if(timer){{clearInterval(timer);timer=null;e.target.textContent='Play';return}}e.target.textContent='Pause';timer=setInterval(()=>{{if(+slider.value>=maxStep){{clearInterval(timer);timer=null;e.target.textContent='Play';return}}slider.value=+slider.value+1;render()}},650)}};window.onresize=render;render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an interactive replay from saved EB-Navigation records.")
    parser.add_argument("--s2m-run", type=Path, required=True, help="S2M result directory")
    parser.add_argument("--baseline-run", type=Path, required=True, help="EB-prompt result directory")
    parser.add_argument("--subset", required=True, help="formal subset directory, e.g. long_horizon")
    parser.add_argument("--episode", type=int, required=True, help="shared episode number, e.g. 42")
    parser.add_argument("--output", type=Path, required=True, help="HTML file to create")
    args = parser.parse_args()

    s2m_path = find_episode(args.s2m_run, args.subset, args.episode)
    baseline_path = find_episode(args.baseline_run, args.subset, args.episode)
    s2m = prepare(read_record(s2m_path), "S2M")
    baseline = prepare(read_record(baseline_path), "EB prompt baseline")
    if not s2m["source_has_coordinates"] or not baseline["source_has_coordinates"]:
        raise ValueError("Both selected records need trajectory x/z coordinates for a spatial replay.")
    payload = {"episode": args.episode, "s2m": s2m, "baseline": baseline,
               "sources": {"s2m": str(s2m_path), "baseline": str(baseline_path)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"S2M source: {s2m_path}")
    print(f"Baseline source: {baseline_path}")


if __name__ == "__main__":
    main()
