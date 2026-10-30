"""Live browser view of S2M: watch the agent while it queries the local VLM, and record every run.

  python demo/live_view.py                 then open http://127.0.0.1:8765
  python demo/live_view.py --host 0.0.0.0  let others on the local network watch; anyone who can reach the port
                                           can also start runs, so use this only on a trusted network

The page streams the rendered frames (first-person view, map, the VLM's answers, the controller's decision) as
they are produced, lists every question sent to the VLM with its answer and latency, and lets you pick an
episode and start or stop a run. Every run is saved to demo/videos/live_*.mp4.
Needs the VLM server: bash demo/run_demo.sh serve (or use bash demo/run_demo.sh show, which starts it).
"""
import argparse
import functools
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import viz  # noqa: E402
from live import check_server, load_episodes, run_live  # noqa: E402
from replay import MODELS, RUNS, SUBSETS, make_env  # noqa: E402

VIDEOS = os.path.join(HERE, "videos")
EB_DATA = os.path.join(ROOT, "third_party", "EmbodiedBench", "embodiedbench", "envs", "eb_navigation", "datasets")
VIDEO_NAME = re.compile(r"^[\w.-]+\.mp4$")
QA = {"zh": {"spec": "把指令改写成要找的目标：「{p}」", "presence": "画面里有「{p}」吗？", "ground": "「{p}」在画面的哪里？",
             "zoom": "放大复核：这个框里是「{p}」吗？", "tile": "切块搜索：这一块里有「{p}」吗？",
             "yes": "P(是) = {:.2f}", "box": "框 {}", "none": "没有", "found": "找到"},
      "en": {"spec": "Rewrite the instruction into a target: “{p}”", "presence": "Is there a {p} in the image?",
             "ground": "Where is the {p}?", "zoom": "Crop check: is this box a {p}?", "tile": "Tiled search: a {p} here?",
             "yes": "P(yes) = {:.2f}", "box": "box {}", "none": "none", "found": "found"}}


class Hub:
    """State shared by the agent loop (main thread) and the HTTP handler threads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.seq = b"", 0
        self.lock = threading.Lock()
        self.calls, self.call_seq, self.step = [], 0, 0
        self.state = {"status": "idle", "run_id": 0}
        self.requests = queue.Queue(maxsize=1)
        self.stop = threading.Event()

    def frame(self, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=85)
        with self.cond:
            self.jpeg, self.seq = buf.getvalue(), self.seq + 1
            self.cond.notify_all()

    def call(self, kind, question, answer, seconds):
        with self.lock:
            self.call_seq += 1
            self.calls.append({"n": self.call_seq, "step": self.step + 1, "kind": kind, "q": question, "a": answer,
                               "ms": round(1000 * seconds)})
            del self.calls[:-500]

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def snapshot(self, since):
        with self.lock:
            return dict(self.state, calls=[c for c in self.calls if c["n"] > since], last_call=self.call_seq)


def instrument(planner, hub, lang):
    """Log every VLM call of the planner's grounder (question, answer, latency) without changing its behavior."""
    g, L = planner.g, QA[lang]

    def box(b):
        return L["box"].format("[" + ", ".join(str(int(v)) for v in b) + "]")

    # method -> (kind, question from the call's args and result, answer from the result)
    specs = {"target_spec": ("spec", lambda a, r: L["spec"].format(p=r.get("query", "")), lambda r: "named" if r.get("named") else "appearance"),
             "presence": ("presence", lambda a, r: L["presence"].format(p=a[1]), lambda r: L["yes"].format(r)),
             "ground": ("ground", lambda a, r: L["ground"].format(p=a[1]), lambda r: box(r[0]) if r and r[0] is not None else L["none"]),
             "zoom_presence": ("zoom", lambda a, r: L["zoom"].format(p=a[2]), lambda r: L["yes"].format(r)),
             "ground_tile": ("tile", lambda a, r: L["tile"].format(p=a[1]), lambda r: L["found"] if r is not None else L["none"])}
    depth = threading.local()  # zoom_presence and ground_tile call presence / ground: log only the outer call
    for name, (kind, question, answer) in specs.items():
        f = getattr(g, name)

        def wrapped(*a, _f=f, _kind=kind, _q=question, _a=answer, **kw):
            outer = getattr(depth, "n", 0) == 0
            depth.n = getattr(depth, "n", 0) + 1
            t0 = time.time()
            try:
                r = _f(*a, **kw)
            finally:
                depth.n -= 1
            if outer:
                try:
                    hub.call(_kind, _q(a, r), _a(r), time.time() - t0)
                except Exception:  # logging must never break the agent
                    pass
            return r
        setattr(g, name, wrapped)


@functools.lru_cache(maxsize=32)
def episodes(split, subset, model):
    """Episode list for the page, with the saved paper outcome of both agents on the test split."""
    path = os.path.join(ROOT, "data", "eb_nav_dev", f"{subset}.json") if split == "dev" else \
        os.path.join(EB_DATA, f"{subset}.json")
    out = []
    for i, t in enumerate(json.load(open(path))["tasks"]):
        rec = {}
        for a in ("s2m", "eb"):
            f = os.path.join(ROOT, "runs", "eb_nav", "test", RUNS[a].format(model=model), subset, f"episode_{i + 1}.json")
            rec[a] = (json.load(open(f))["task_success"] >= 1) if split == "test" and os.path.exists(f) else None
        out.append({"n": i + 1, "instruction": t["instruction"], "s2m": rec["s2m"], "eb": rec["eb"]})
    return out


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2M 实时演示</title>
<style>
:root{--page:#f4f3f0;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#96958f;--line:#dcdbd6;
--blue:#2a78d6;--orange:#eb6834;--good:#008300;--bad:#e34948}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:12px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0;font-weight:700}header .sub{color:var(--ink2);font-size:13px}
.chip{margin-left:auto;padding:3px 10px;border-radius:999px;font-weight:600;font-size:13px;background:#eceae6;color:var(--ink2)}
.chip.running{background:#dce9f8;color:var(--blue)}.chip.done{background:#ddeedd;color:var(--good)}
.chip.error,.chip.stopped{background:#f8e1e0;color:var(--bad)}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px 16px}
@media (max-width:1100px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.stream img{width:100%;height:auto;display:block;border-radius:4px;background:#ddd}
label{display:block;font-size:12px;color:var(--ink2);margin:8px 0 3px}
select,input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff;color:var(--ink)}
.row{display:flex;gap:8px}.row>*{flex:1}
button{padding:8px 12px;border-radius:6px;border:1px solid var(--line);font:inherit;font-weight:600;cursor:pointer;background:#fff;color:var(--ink)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button:disabled{opacity:.45;cursor:default}
.instr{font-size:13px;color:var(--ink2);margin-top:6px;min-height:2.6em}
.badge{display:inline-block;font-size:12px;padding:0 6px;border-radius:4px;margin-right:4px;border:1px solid var(--line)}
.stat{font-size:13px;margin-top:8px;color:var(--ink2)}.stat b{color:var(--ink)}
h2{font-size:14px;margin:0 0 8px}
#log{height:380px;overflow:auto;font-size:12.5px;border-top:1px solid var(--line);padding-top:6px}
.call{padding:4px 0;border-bottom:1px solid #f0efeb}.call .q{color:var(--ink)}
.call .a{color:var(--blue);font-weight:600}.call .m{color:var(--muted);font-size:11.5px}
.vids a{display:block;font-size:12.5px;color:var(--blue);text-decoration:none;padding:2px 0;word-break:break-all}
.hint{font-size:12px;color:var(--muted);margin-top:8px}
</style></head><body>
<header><h1>S2M 实时演示</h1><span class="sub">小模型只回答"目标在不在画面里、在哪里"，几何控制器负责记忆和行动</span>
<span id="chip" class="chip">空闲</span></header>
<main>
<section class="card stream"><img id="live" src="/stream" alt="实时画面"></section>
<aside>
<div class="card">
<h2>选择回合</h2>
<div class="row"><div><label for="split">数据集</label><select id="split"><option value="test">测试集</option><option value="dev">开发集</option></select></div>
<div><label for="subset">子集</label><select id="subset"></select></div></div>
<label for="episode">回合</label><select id="episode"></select>
<div class="instr" id="instr"></div>
<label for="delay">每步额外停顿（秒），方便观众跟上</label><input id="delay" type="number" min="0" max="5" step="0.5" value="1">
<div class="row" style="margin-top:10px"><button class="primary" id="run">开始运行</button><button id="stop" disabled>停止</button></div>
<div class="stat" id="stat"></div>
<div class="hint">每次运行都会自动录成 MP4，保存在 demo/videos/。</div>
</div>
<div class="card" style="margin-top:12px"><h2>发给小模型的每一个问题</h2><div id="log"></div></div>
<div class="card vids" style="margin-top:12px"><h2>已录制的视频</h2><div id="vids"></div></div>
</aside></main>
<script>
const SUBSETS={base:"Base（直接给类别名）",common_sense:"Common-sense（用途描述）",complex_instruction:"Complex（长指令）",
visual_appearance:"Visual（只有外观）",long_horizon:"Long-horizon（目标在身后）"};
const STATUS={idle:"空闲",starting:"准备中",running:"运行中",done:"已完成",stopped:"已停止",error:"出错"};
const $=id=>document.getElementById(id);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
for(const [k,v] of Object.entries(SUBSETS)){const o=document.createElement("option");o.value=k;o.textContent=v;$("subset").appendChild(o);}
let eps=[],runId=0,lastCall=0;
function mark(v,name){return v===null?"":`<span class="badge">${name} ${v?"✓":"✕"}</span>`}
async function loadEpisodes(){
  const r=await fetch(`/episodes?split=${$("split").value}&subset=${$("subset").value}`);eps=await r.json();
  $("episode").innerHTML="";
  for(const e of eps){const o=document.createElement("option");o.value=e.n;
    o.textContent=`#${e.n}  ${e.s2m===null?"":(e.s2m?"S2M✓":"S2M✕")+" "+(e.eb?"EB✓":"EB✕")}  ${e.instruction}`;$("episode").appendChild(o);}
  showInstr();}
function showInstr(){const e=eps.find(x=>x.n==$("episode").value);if(!e){$("instr").textContent="";return;}
  $("instr").textContent="";const t=document.createElement("div");t.textContent="“"+e.instruction+"”";$("instr").appendChild(t);
  if(e.s2m!==null){const s=document.createElement("div");s.innerHTML="论文记录："+mark(e.s2m,"S2M")+mark(e.eb,"EB prompt");$("instr").appendChild(s);}}
$("split").onchange=loadEpisodes;$("subset").onchange=loadEpisodes;$("episode").onchange=showInstr;
$("run").onclick=async()=>{const body={split:$("split").value,subset:$("subset").value,episode:+$("episode").value,delay:+$("delay").value};
  const r=await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){alert((await r.json()).error||"无法开始");}};
$("stop").onclick=()=>fetch("/stop",{method:"POST"});
function addCall(c){const d=document.createElement("div");d.className="call";
  const q=document.createElement("div");q.className="q";q.textContent=`第${c.step}步 · ${c.q}`;
  const a=document.createElement("div");a.className="a";a.textContent="→ "+c.a;
  const m=document.createElement("div");m.className="m";m.textContent=`${c.ms} ms`;
  d.append(q,a,m);$("log").appendChild(d);}
async function poll(){
  try{const s=await (await fetch(`/status?since=${lastCall}`)).json();
    if(s.run_id!==runId){runId=s.run_id;lastCall=0;$("log").innerHTML="";}
    for(const c of s.calls){if(c.n>lastCall){addCall(c);lastCall=c.n;}}
    if(s.calls.length){$("log").scrollTop=$("log").scrollHeight;}
    $("chip").textContent=STATUS[s.status]||s.status;$("chip").className="chip "+s.status;
    const busy=s.status==="running"||s.status==="starting";$("run").disabled=busy;$("stop").disabled=!busy;
    let t="";if(s.episode)t+=`<b>${esc(SUBSETS[s.subset]||s.subset)} #${esc(s.episode)}</b>（${s.split==="dev"?"开发集":"测试集"}）<br>`;
    if(s.step!==undefined)t+=`第 <b>${esc(s.step)}</b> 步 · ${esc(s.action)} ${s.ok===false?"（被拒绝）":""}<br>`;
    if(s.distance!==undefined)t+=`距目标 <b>${esc(s.distance)} m</b> · 已调用模型 <b>${esc(s.vlm_calls||0)}</b> 次<br>`;
    if(s.result)t+=`<b style="color:${s.result.success?"var(--good)":"var(--bad)"}">${s.result.success?"✓ 成功":(s.result.aborted?"已停止":"✕ 失败")}</b>，${esc(s.result.steps)} 步`;
    if(s.error)t+=`<span style="color:var(--bad)">${esc(s.error)}</span>`;
    $("stat").innerHTML=t;
    if(s.videos){const v=$("vids");v.innerHTML="";for(const n of s.videos){const a=document.createElement("a");a.href="/video/"+n;a.textContent=n;a.target="_blank";v.appendChild(a);}}
  }catch(e){}
  setTimeout(poll,500);}
loadEpisodes();poll();
</script></body></html>
"""


def make_handler(hub, model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with hub.cond:
                            hub.cond.wait_for(lambda: hub.seq != seen, timeout=10)
                            jpg, seen = hub.jpeg, hub.seq
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif u.path == "/status":
                try:
                    since = int(q.get("since", "0"))
                except ValueError:
                    since = 0
                s = hub.snapshot(since)
                s["videos"] = sorted((f for f in os.listdir(VIDEOS) if f.startswith("live_") and f.endswith(".mp4")),
                                     reverse=True)[:12]
                self._json(s)
            elif u.path == "/episodes":
                split, subset = q.get("split", "test"), q.get("subset", "base")
                if split not in ("test", "dev") or subset not in SUBSETS:
                    return self._json({"error": "bad split or subset"}, 400)
                self._json(episodes(split, subset, model))
            elif u.path.startswith("/video/"):
                name = os.path.basename(urllib.parse.unquote(u.path[len("/video/"):]))
                path = os.path.join(VIDEOS, name)
                if not VIDEO_NAME.match(name) or not os.path.isfile(path):
                    return self._json({"error": "not found"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(path)))
                self.end_headers()
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/stop":
                hub.stop.set()
                return self._json({"ok": True})
            if u.path != "/run":
                return self._json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                split, subset = req.get("split"), req.get("subset")
                ep, delay = int(req.get("episode")), float(req.get("delay", 1.0))
            except Exception:
                return self._json({"error": "bad request"}, 400)
            if split not in ("test", "dev") or subset not in SUBSETS or not 1 <= ep <= 60 or not 0 <= delay <= 5:
                return self._json({"error": "bad split, subset, episode or delay"}, 400)
            if hub.state.get("status") in ("starting", "running"):
                return self._json({"error": "已有一个回合在运行"}, 409)
            try:
                hub.requests.put_nowait({"split": split, "subset": subset, "episode": ep, "delay": delay})
            except queue.Full:
                return self._json({"error": "已有一个回合在排队"}, 409)
            hub.update(status="starting")
            return self._json({"ok": True})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let others on the network watch (trusted networks only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="qwen3", choices=list(MODELS))
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="language of the frames and the call log")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()
    model = MODELS[args.model]
    err = check_server(model)
    if err:
        sys.exit(err)
    os.makedirs(VIDEOS, exist_ok=True)
    from embagent.planners import make_planner
    hub = Hub()
    env, _, skills = make_env("base")
    planner = make_planner("s2m", skills, model_name=model, s2m={})
    instrument(planner, hub, args.lang)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, model))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    idle = {"zh": (("S2M 实时演示", "在右侧选择一个回合并点击「开始运行」"), "画面会在智能体运行时实时更新；每次运行都会录成视频。"),
            "en": (("S2M live demo", "Pick an episode on the right and press Start"), "The view updates while the agent runs; every run is recorded.")}
    lines, sub = idle[args.lang]
    hub.frame(viz.title_card(lines, sub, args.lang))
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"live view at {url}" + ("  (also reachable from other machines on this network)" if args.host == "0.0.0.0" else ""),
          flush=True)
    if args.open and os.environ.get("DISPLAY"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while True:
            req = hub.requests.get()
            hub.stop.clear()
            with hub.lock:
                hub.calls, hub.step = [], 0
            hub.update(status="running", run_id=hub.state["run_id"] + 1, result=None, error=None, step=0,
                       action="", ok=None, distance=None, vlm_calls=0, **req)
            err = check_server(model)
            if err:
                hub.update(status="error", error=err)
                continue
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = os.path.join(VIDEOS, f"live_{stamp}_{req['split']}_{req['subset']}_ep{req['episode']}_{args.lang}.mp4")
            try:
                full = load_episodes(env, req["split"], req["subset"])

                def on_step(s):
                    with hub.lock:
                        hub.step = s["step"]
                    hub.update(step=s["step"], action=s["action"], ok=s["ok"], distance=s["distance"],
                               vlm_calls=planner.g.calls)

                r = run_live(env, full, req["episode"] - 1, planner, skills, model, args.lang, args.fps, out=out,
                             delay=req["delay"], on_frame=hub.frame, on_step=on_step, should_stop=hub.stop.is_set)
                hub.update(status="stopped" if r["aborted"] else "done", result=r, vlm_calls=r["vlm_calls"])
                print(f"{req}: {'stopped' if r['aborted'] else ('success' if r['success'] else 'failed')} after "
                      f"{r['steps']} steps; {os.path.basename(out)}", flush=True)
            except Exception as e:
                traceback.print_exc()
                hub.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
                try:
                    env.env.stop()
                except Exception:
                    pass
                env, _, skills = make_env("base")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            env.env.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
