"""Live side-by-side view: the EB prompt and S2M run the same episode at the same time on the same local VLM.

  python demo/live_view.py                 then open http://127.0.0.1:8765
  python demo/live_view.py --host 0.0.0.0  let others on the local network watch; anyone who can reach the port
                                           can also start runs, so use this only on a trusted network
  python demo/live_view.py --once test:long_horizon:14   run one episode without the browser, record it, exit

Each agent runs in its own process with its own simulator. Its panel (camera view with smooth motion, map, the
model's answers, the decisions) is streamed to the browser as it is produced, and while the agent waits for the
model the panel shows what is being asked and for how long. Every run is recorded side by side to
demo/videos/live_*.mp4. Needs the VLM server (bash demo/run_demo.sh show starts it).
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import shared_memory

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import render as R  # noqa: E402

MODELS = {"qwen3": "Qwen3-VL-4B-Instruct-AWQ", "qwen25": "Qwen2.5-VL-3B-Instruct-AWQ"}
RUNS = {"s2m": "s2m_{m}_final_r1", "eb": "eb_{m}_eb_official_r1"}
VIDEOS = os.path.join(HERE, "videos")
VIDEO_NAME = re.compile(r"^[\w.-]+\.mp4$")
RAW = R.PW * R.PH * 3
JPG_MAX = 2 << 20
MIN_CONTEXT = 4608  # EB-prompt requests: up to ~3.9k prompt tokens plus the completion budget


class Slot:
    """Latest panel of one agent in shared memory: raw RGB (for the recording) and a JPEG (for the browser)."""

    def __init__(self, name=None, lock=None, seq=None):
        # workers share the parent's resource tracker, so attaching registers nothing new; the parent unlinks
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=RAW + 4 + JPG_MAX)
        self.lock, self.seq = lock or mp.Lock(), seq or mp.Value("q", 0, lock=False)
        self.raw = np.ndarray((R.PH, R.PW, 3), np.uint8, self.shm.buf[:RAW])

    def write(self, img):
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        jpg = buf.getvalue()[:JPG_MAX]
        with self.lock:
            np.copyto(self.raw, np.asarray(img))
            self.shm.buf[RAW:RAW + 4] = len(jpg).to_bytes(4, "little")
            self.shm.buf[RAW + 4:RAW + 4 + len(jpg)] = jpg
            self.seq.value += 1

    def read_raw(self, out):
        with self.lock:
            np.copyto(out, self.raw)
            return self.seq.value

    def read_jpeg(self):
        with self.lock:
            n = int.from_bytes(bytes(self.shm.buf[RAW:RAW + 4]), "little")
            return self.seq.value, bytes(self.shm.buf[RAW + 4:RAW + 4 + n])


class SlotWriter:
    """Encodes and publishes panels on its own thread, so the paced frame loop only renders and waits."""

    def __init__(self, slot):
        self.slot, self.q = slot, queue.Queue(maxsize=3)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            self.slot.write(self.q.get())

    def write(self, img):
        self.q.put(img)


class Thinking:
    """While a planner waits for the model, keep redrawing its panel with what is being asked and for how long."""

    def __init__(self, slot, fps=6):
        self.slot, self.dt, self.lock = slot, 1.0 / fps, threading.Lock()
        self.thread, self.active, self.question = None, False, ""

    def ask(self, question):
        with self.lock:
            self.question = question

    def start(self, panel, rgb, pose, default):
        self.stop()
        self.active, self.t0, self.question = True, time.time(), default

        def loop():
            while self.active:
                with self.lock:
                    q = self.question
                img = panel.frame(rgb, pose, thinking=(time.time() - self.t0, q))
                if self.active:
                    self.slot.write(img)
                time.sleep(self.dt)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def worker(agent, model, lang, fps, max_len, slot_args, cmd_q, ev_q, stop_ev):
    """One agent in its own process: simulator, planner, panel rendering, paced frame output."""
    os.environ["VLM_MAX_MODEL_LEN"] = str(max_len)
    os.environ["VLM_MAX_TOKENS"] = str(max(512, min(1024, max_len - 3950)))
    os.environ.setdefault("VLM_SCHEMA_BOUNDS", json.dumps({"max_chars": 1500, "max_items": 20, "name_chars": 80}))
    import numpy as np  # noqa: F811
    from agents import EBLive, S2MLive
    from sim import AgentSim, episode
    T = R.TEXT[lang]
    slot = SlotWriter(Slot(*slot_args))
    slot.write(R.placeholder(agent, lang, T["ready"]))
    think = Thinking(slot)

    def on_call(kind, q, a, secs):
        ev_q.put(("call", agent, {"kind": kind, "q": q, "a": a, "ms": round(1000 * secs)}))

    try:
        sim = AgentSim()
        cls = S2MLive if agent == "s2m" else EBLive
        src = cls(model, sim.skills, lang, on_call, think.ask)
    except Exception as e:
        ev_q.put(("error", agent, f"{type(e).__name__}: {e}"[:300]))
        return
    slot.write(R.placeholder(agent, lang, T["idle"]))
    ev_q.put(("ready", agent, None))
    while True:
        cmd = cmd_q.get()
        if cmd[0] == "quit":
            break
        _, split, subset, ep, pause = cmd
        try:
            sim.load(split, subset)
            obs, floor, target, instr = sim.reset(ep - 1)
            view = R.MapView(floor, [target], margin=0.9)
            panel = R.Panel(agent, lang, model, view, sim.budget, target)
            gen = episode(sim, src, panel, obs, instr, fps, pause=pause, stop=stop_ev.is_set)
            deadline, frames, last_motion, gaps = time.time(), 0, None, []
            for item in gen:
                think.stop()
                if item[0] == "think":
                    think.start(item[1], item[2], item[3], T["look"] if agent == "s2m" else T["think_eb"])
                    deadline = None
                elif item[0] == "frame":
                    now = time.time()
                    if deadline is None or now - deadline > 0.25:
                        deadline = now
                    slot.write(item[1])
                    frames += 1
                    t = time.time()
                    if item[2] < 0.1:  # a motion frame: record the interval to the previous one
                        if last_motion is not None and t - last_motion < 0.5:
                            gaps.append(t - last_motion)
                        last_motion = t
                    else:
                        last_motion = None
                    deadline += item[2]
                    wait = deadline - time.time()
                    if wait > 0:
                        time.sleep(wait)
                elif item[0] == "step":
                    ev_q.put(("step", agent, item[1]))
                elif item[0] == "done":
                    g = np.array(gaps) * 1000 if gaps else np.zeros(1)
                    ev_q.put(("done", agent, dict(item[1], frames=frames, motion_ms=[round(float(np.median(g))),
                                                  round(float(np.percentile(g, 95))), round(float(g.max()))])))
        except Exception as e:
            think.stop()
            traceback.print_exc()
            ev_q.put(("error", agent, f"{type(e).__name__}: {e}"[:300]))
            ev_q.put(("done", agent, {"error": True}))
            sim.close()
            sim = AgentSim()
    think.stop()
    sim.close()


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2M 与 EB prompt 实时对比</title>
<style>
:root{--page:#f4f3f0;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#96958f;--line:#dcdbd6;
--blue:#2a78d6;--orange:#eb6834;--good:#008300;--bad:#e34948}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:16px;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0;font-weight:700;white-space:nowrap}
#ep{color:var(--ink2);font-size:13px;white-space:nowrap}#instr{font-weight:600;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip{padding:3px 10px;border-radius:999px;font-weight:600;font-size:13px;background:#eceae6;color:var(--ink2);white-space:nowrap}
.chip.running,.chip.starting{background:#dce9f8;color:var(--blue)}.chip.done{background:#ddeedd;color:var(--good)}
.chip.error,.chip.stopped{background:#f8e1e0;color:var(--bad)}
.streams{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--line)}
.streams img{width:100%;height:auto;display:block;background:var(--page)}
.bar{display:grid;grid-template-columns:330px 1fr 1fr;gap:12px;padding:12px 16px}
@media (max-width:1100px){.bar{grid-template-columns:1fr}.streams{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
label{display:block;font-size:12px;color:var(--ink2);margin:6px 0 3px}
select,input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff;color:var(--ink)}
.row{display:flex;gap:8px}.row>*{flex:1}
button{padding:8px 12px;border-radius:6px;border:1px solid var(--line);font:inherit;font-weight:600;cursor:pointer;background:#fff;color:var(--ink)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button:disabled{opacity:.45;cursor:default}
.hint{font-size:12px;color:var(--muted);margin-top:8px}
h2{font-size:14px;margin:0 0 6px;display:flex;align-items:center;gap:8px}
h2 .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.stat{font-size:13px;color:var(--ink2);margin-bottom:6px;min-height:2.9em}.stat b{color:var(--ink)}
.log{height:300px;overflow:auto;font-size:12.5px;border-top:1px solid var(--line);padding-top:4px}
.call{padding:3px 0;border-bottom:1px solid #f0efeb}.call .a{font-weight:600}.call .m{color:var(--muted);font-size:11.5px}
.eb .call .a{color:var(--orange)}.s2m .call .a{color:var(--blue)}
.vids a{display:block;font-size:12px;color:var(--blue);text-decoration:none;word-break:break-all}
</style></head><body>
<header><h1>同一个 4B 小模型的两种用法</h1><span id="ep"></span><span id="instr"></span><span id="chip" class="chip">准备中</span></header>
<div class="streams"><img src="/stream/eb" alt="EB prompt 实时画面"><img src="/stream/s2m" alt="S2M 实时画面"></div>
<div class="bar">
<div class="card">
<h2>选择回合</h2>
<div class="row"><div><label for="split">数据集</label><select id="split"><option value="test">测试集</option><option value="dev">开发集</option></select></div>
<div><label for="subset">子集</label><select id="subset"></select></div></div>
<label for="episode">回合（✓/× 为论文记录里的结果）</label><select id="episode"></select>
<label for="pause">每次决策后停顿（秒），方便观众看清</label><input id="pause" type="number" min="0" max="5" step="0.2" value="0.6">
<div class="row" style="margin-top:10px"><button class="primary" id="run">开始运行</button><button id="stop" disabled>停止</button></div>
<div class="hint">两个智能体同时运行、共用同一个本地模型。EB prompt 每次要生成一整段 JSON 计划，所以“思考”时间明显更长。每次运行自动录成并排视频。</div>
<div class="vids" style="margin-top:8px" id="vids"></div>
</div>
<div class="card eb"><h2><span class="dot" style="background:var(--orange)"></span>EB prompt：模型的每一次调用</h2><div class="stat" id="stat_eb"></div><div class="log" id="log_eb"></div></div>
<div class="card s2m"><h2><span class="dot" style="background:var(--blue)"></span>S2M：模型的每一次调用</h2><div class="stat" id="stat_s2m"></div><div class="log" id="log_s2m"></div></div>
</div>
<script>
const SUBSETS={base:"Base（直接给类别）",common_sense:"Common-sense（用途描述）",complex_instruction:"Complex（长指令）",
visual_appearance:"Visual（只有外观）",long_horizon:"Long-horizon（目标在身后）"};
const STATUS={idle:"空闲",preparing:"准备中",starting:"启动中",running:"运行中",done:"已完成",stopped:"已停止",error:"出错"};
const $=id=>document.getElementById(id);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
for(const [k,v] of Object.entries(SUBSETS)){const o=document.createElement("option");o.value=k;o.textContent=v;$("subset").appendChild(o);}
let runId=-1,lastCall=0;
async function loadEpisodes(){
  const eps=await (await fetch(`/episodes?split=${$("split").value}&subset=${$("subset").value}`)).json();$("episode").innerHTML="";
  for(const e of eps){const o=document.createElement("option");o.value=e.n;
    const m=e.s2m===null?"":`S2M${e.s2m?"✓":"×"} EB${e.eb?"✓":"×"}  `;o.textContent=`#${e.n}  ${m}${e.instruction}`;$("episode").appendChild(o);}}
$("split").onchange=loadEpisodes;$("subset").onchange=loadEpisodes;
$("run").onclick=async()=>{const body={split:$("split").value,subset:$("subset").value,episode:+$("episode").value,pause:+$("pause").value};
  const r=await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){alert((await r.json()).error||"无法开始");}};
$("stop").onclick=()=>fetch("/stop",{method:"POST"});
function addCall(c){const d=document.createElement("div");d.className="call";
  const q=document.createElement("div");q.textContent=c.q;const a=document.createElement("div");a.className="a";a.textContent="→ "+c.a;
  const m=document.createElement("div");m.className="m";m.textContent=`${(c.ms/1000).toFixed(2)} s`;d.append(q,a,m);
  const log=$("log_"+c.agent);log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function poll(){
  try{const s=await (await fetch(`/status?since=${lastCall}`)).json();
    if(s.run_id!==runId){runId=s.run_id;lastCall=0;$("log_eb").innerHTML="";$("log_s2m").innerHTML="";}
    for(const c of s.calls){if(c.n>lastCall){addCall(c);lastCall=c.n;}}
    $("chip").textContent=STATUS[s.status]||s.status;$("chip").className="chip "+s.status;
    $("ep").textContent=s.episode?`${s.split==="dev"?"开发集":"测试集"} · ${SUBSETS[s.subset]} · #${s.episode}`:"";
    $("instr").textContent=s.instruction?`“${s.instruction}”`:"";
    const busy=["starting","running","preparing"].includes(s.status);$("run").disabled=busy;$("stop").disabled=!(s.status==="running"||s.status==="starting");
    for(const a of ["eb","s2m"]){const g=s.agents[a]||{};let t="";
      if(g.step!==undefined)t+=`第 <b>${esc(g.step)}</b> 步 · ${esc(g.action||"")}${g.ok===false?"（被拒绝）":""}<br>`;
      if(g.distance!==undefined)t+=`距目标 <b>${esc(g.distance)} m</b> · 调用模型 <b>${esc(g.calls||0)}</b> 次<br>`;
      if(g.result){const r=g.result;t+=r.error?`<b style="color:var(--bad)">出错</b>`:(r.aborted?"<b>已停止</b>":
        `<b style="color:${r.success?"var(--good)":"var(--bad)"}">${r.success?"✓ 成功":"× 失败"}</b>，${esc(r.steps)} 步`);}
      if(g.error)t+=` <span style="color:var(--bad)">${esc(g.error)}</span>`;
      $("stat_"+a).innerHTML=t;}
    const v=$("vids");v.innerHTML="";for(const n of s.videos){const l=document.createElement("a");l.href="/video/"+n;l.textContent=n;l.target="_blank";v.appendChild(l);}
  }catch(e){}
  setTimeout(poll,500);}
loadEpisodes();poll();
</script></body></html>
"""


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "preparing", "run_id": 0, "agents": {a: {} for a in R.AGENTS}}
        self.calls, self.call_seq = [], 0
        self.requests = queue.Queue(maxsize=1)
        self.slots = {}

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def agent(self, a, **kw):
        with self.lock:
            self.state["agents"][a].update(kw)

    def call(self, a, c):
        with self.lock:
            self.call_seq += 1
            self.calls.append(dict(c, agent=a, n=self.call_seq))
            del self.calls[:-800]

    def snapshot(self, since):
        with self.lock:
            s = json.loads(json.dumps(self.state))
            s["calls"] = [c for c in self.calls if c["n"] > since]
        s["videos"] = sorted((f for f in os.listdir(VIDEOS) if f.startswith("live_") and f.endswith(".mp4")),
                             reverse=True)[:10]
        return s


@functools.lru_cache(maxsize=32)
def episodes(split, subset, model):
    from sim import dataset_path
    out = []
    for i, t in enumerate(json.load(open(dataset_path(split, subset)))["tasks"]):
        r = {}
        for a in R.AGENTS:
            f = os.path.join(ROOT, "runs", "eb_nav", "test", RUNS[a].format(m=model), subset, f"episode_{i + 1}.json")
            r[a] = (json.load(open(f))["task_success"] >= 1) if split == "test" and os.path.exists(f) else None
        out.append({"n": i + 1, "instruction": t["instruction"], "s2m": r["s2m"], "eb": r["eb"]})
    return out


def make_handler(hub, model):
    from sim import SUBSETS

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
            elif u.path in ("/stream/eb", "/stream/s2m"):
                slot = hub.slots[u.path.rsplit("/", 1)[1]]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        seq = slot.seq.value
                        if seq != seen:
                            seen, jpg = slot.read_jpeg()
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                                self.wfile.flush()
                        time.sleep(0.012)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif u.path == "/status":
                try:
                    since = int(q.get("since", "0"))
                except ValueError:
                    since = 0
                self._json(hub.snapshot(since))
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
                req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 4096)) or b"{}")
                split, subset, ep, pause = req["split"], req["subset"], int(req["episode"]), float(req.get("pause", .6))
            except Exception:
                return self._json({"error": "bad request"}, 400)
            if split not in ("test", "dev") or subset not in SUBSETS or not 1 <= ep <= 60 or not 0 <= pause <= 5:
                return self._json({"error": "bad split, subset, episode or pause"}, 400)
            if hub.state["status"] != "idle" and hub.state["status"] not in ("done", "stopped", "error"):
                return self._json({"error": "已有一个回合在运行或模拟器还没准备好"}, 409)
            try:
                hub.requests.put_nowait((split, subset, ep, pause))
            except queue.Full:
                return self._json({"error": "已有一个回合在排队"}, 409)
            hub.update(status="starting")
            return self._json({"ok": True})
    return Handler


class Recorder:
    """Writes composited frames to an MP4 from a thread, so a slow encoder never stalls the compositor."""

    def __init__(self, path, fps):
        self.w, self.q = R.VideoWriter(path, fps=fps, preset="veryfast"), queue.Queue(maxsize=90)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.w.write(item[0], repeat=item[1])

    def put(self, frame, repeat=1):
        self.q.put((frame, repeat))

    def close(self):
        self.q.put(None)
        self.t.join()
        return self.w.close()


def server_context(model):
    url = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        data = json.loads(urllib.request.urlopen(url + "/models", timeout=5).read())
    except Exception:
        return None, f"No VLM server at {url}. Start it first: bash demo/run_demo.sh show"
    for m in data.get("data", []):
        if m.get("id") == model:
            n = int(m.get("max_model_len") or 0)
            if n < MIN_CONTEXT:
                return None, (f"The server's context ({n} tokens) is too small for the EB prompt (needs {MIN_CONTEXT}). "
                              "Restart it: bash demo/run_demo.sh stop && bash demo/run_demo.sh show")
            return n, None
    return None, f"The server at {url} does not serve {model}. Restart it: bash demo/run_demo.sh show"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0: reachable from the local network (trusted networks only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="qwen3", choices=list(MODELS))
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--fps", type=int, default=15, help="frames per second of each agent's motion")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--once", default="", help="SPLIT:SUBSET:EPISODE: run one episode, record it, and exit")
    args = ap.parse_args()
    model = MODELS[args.model]
    max_len, err = server_context(model)
    if err:
        sys.exit(err)
    os.makedirs(VIDEOS, exist_ok=True)
    ctx, hub = mp.get_context("spawn"), Hub()
    hub.stop, ev_q = ctx.Event(), ctx.Queue()
    procs, cmd_qs = {}, {}
    for a in R.AGENTS:
        lock, seq = ctx.Lock(), ctx.Value("q", 0, lock=False)
        hub.slots[a] = Slot(None, lock, seq)
        cmd_qs[a] = ctx.Queue()
        procs[a] = ctx.Process(target=worker, daemon=True, args=(a, model, args.lang, args.fps, max_len,
                                                                 (hub.slots[a].shm.name, lock, seq), cmd_qs[a], ev_q, hub.stop))
        procs[a].start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, model))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"live view at {url}" + ("  (also reachable from other machines on this network)" if args.host == "0.0.0.0" else ""),
          flush=True)
    if args.open and os.environ.get("DISPLAY"):
        import subprocess
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if args.once:
        split, subset, ep = args.once.split(":")
        hub.requests.put((split, subset, int(ep), 0.6))
    vfps, ready, busy, rec, tail, ran, out, stats = 30, set(), set(), None, 0, False, None, {}
    canvas = np.zeros((R.H, R.W, 3), np.uint8)
    t_next = time.time()
    try:
        while True:
            while True:
                try:
                    kind, a, data = ev_q.get_nowait()
                except queue.Empty:
                    break
                if kind == "ready":
                    ready.add(a)
                    if ready == set(R.AGENTS) and hub.state["status"] == "preparing":
                        hub.update(status="idle")
                elif kind == "call":
                    hub.call(a, data)
                elif kind == "step":
                    hub.agent(a, step=data["step"], action=data["action"], ok=data["ok"], distance=data["distance"],
                              calls=data["calls"])
                elif kind == "error":
                    hub.agent(a, error=data)
                    print(f"[{a}] error: {data}", flush=True)
                elif kind == "done":
                    busy.discard(a)
                    stats[a] = data
                    hub.agent(a, result=data, calls=data.get("calls", hub.state["agents"][a].get("calls")))
            for a, p in procs.items():
                if not p.is_alive() and (a in busy or a not in ready):
                    hub.update(status="error", error=f"{a} worker exited")
                    busy.discard(a)
                    ready.discard(a)
            if rec is None and ready == set(R.AGENTS) and not hub.requests.empty():
                split, subset, ep, pause = hub.requests.get()
                from sim import dataset_path
                instr = json.load(open(dataset_path(split, subset)))["tasks"][ep - 1]["instruction"]
                hub.stop.clear()
                with hub.lock:
                    hub.calls = []
                hub.update(status="running", run_id=hub.state["run_id"] + 1, split=split, subset=subset, episode=ep,
                           instruction=instr, error=None, agents={a: {} for a in R.AGENTS})
                out = os.path.join(VIDEOS, f"live_{time.strftime('%Y%m%d-%H%M%S')}_{split}_{subset}_ep{ep}_{args.lang}.mp4")
                rec = Recorder(out, vfps)
                canvas[:R.HEAD] = np.asarray(R.header(args.lang, "live", split, subset, ep, instr))
                busy, tail, ran, stats = set(R.AGENTS), 0, True, {}
                for a in R.AGENTS:
                    cmd_qs[a].put(("run", split, subset, ep, pause))
            now = time.time()
            if rec is not None and now >= t_next:
                n = 1 + int((now - t_next) * vfps)  # keep the recording in real time if a tick was late
                for i, a in enumerate(R.AGENTS):
                    hub.slots[a].read_raw(canvas[R.HEAD:, i * R.PW:(i + 1) * R.PW])
                canvas[R.HEAD:, R.PW - 1:R.PW + 1] = R.LINE
                rec.put(canvas.copy(), repeat=n)
                t_next += n / vfps
                if not busy:
                    tail += n
                    if tail >= vfps:  # one more second after both agents finished
                        rec.close()
                        rec = None
                        status = "stopped" if hub.stop.is_set() else ("error" if any(s.get("error") for s in stats.values()) else "done")
                        hub.update(status=status)
                        print(f"{status}: " + "; ".join(
                            f"{a} success={stats.get(a, {}).get('success')} steps={stats.get(a, {}).get('steps')} "
                            f"calls={stats.get(a, {}).get('calls')} frames={stats.get(a, {}).get('frames')} "
                            f"motion interval ms median/p95/max={stats.get(a, {}).get('motion_ms')}" for a in R.AGENTS) + f"; video {out}", flush=True)
                        if args.once:
                            break
            elif rec is None:
                t_next = now
            time.sleep(0.004)
            if args.once and ran and rec is None and hub.state["status"] in ("error",) and not busy:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None:
            rec.close()
        for a in R.AGENTS:
            cmd_qs[a].put(("quit",))
        for p in procs.values():
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        server.shutdown()
        for s in hub.slots.values():
            s.shm.close()
            s.shm.unlink()


if __name__ == "__main__":
    main()
