"""Frame rendering for the S2M / EB-prompt demo (live view and replay videos).

A 1920x1080 frame is a shared header plus two 960x996 panels, one per agent (EB prompt left, S2M right). Each
panel has a static layer, redrawn when the agent decides or acts (title, step, timeline, the model's answers, the
controller's decisions, map content), and a cheap dynamic layer drawn for every frame of smooth motion (camera
view, agent marker, field of view, thinking and outcome overlays).
"""
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, HEAD = 1920, 1080, 84
PW, PH = W // 2, H - HEAD                    # one agent panel
IMG, IMG_Y, FPV_X, MAP_X = 448, 80, 20, 492  # square camera view and map
TL_Y = IMG_Y + IMG + 14                      # timeline: one cell per step
TX_Y, TX_W = TL_Y + 42, 448                  # two text columns at FPV_X and MAP_X
LEG_Y = PH - 32
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)
INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (150, 149, 144)
PANEL, PAGE, LINE = (255, 255, 255), (244, 243, 240), (220, 219, 214)
FLOOR, SEEN, OCC = (229, 228, 224), (205, 222, 243), (96, 95, 92)
GOOD, BAD = (0, 131, 0), (227, 73, 72)
FONT_DIR = "/usr/share/fonts/opentype/noto"
AGENTS = ("eb", "s2m")  # left, right

TEXT = {
    "zh": {
        "title": "同一个 4B 小模型的两种用法", "sep": "，", "split": {"test": "测试集", "dev": "开发集"},
        "subsets": {"base": "Base：直接给出类别", "common_sense": "Common-sense：用途描述",
                    "complex_instruction": "Complex：长指令", "visual_appearance": "Visual：只有外观描述",
                    "long_horizon": "Long-horizon：目标在身后"},
        "note_live": "实时运行：两个智能体同时调用同一个本地模型。星号是真实目标，只给观众看，智能体看不到。",
        "note_replay": "回放：动作、模型回答和决策来自论文测试集的保存记录。星号是真实目标，只给观众看。",
        "s2m": "S2M（本文方法）", "eb": "EB prompt（基准自带的用法）",
        "s2m_sub": "小模型只回答“目标在不在、在哪”，几何控制器负责记忆和行动",
        "eb_sub": "小模型看图后直接输出多步动作计划（JSON）",
        "fpv": "智能体看到的画面", "map": "俯视地图：S2M 自己建的地图和目标假设",
        "map_eb": "俯视图：只给观众看，EB prompt 不维护地图",
        "vlm": "小模型的回答", "dec": "几何控制器的决策", "vlm_eb": "小模型输出的计划", "dec_eb": "执行情况",
        "step": "第 {} 步", "of": " / 上限 {}", "pyes": "P(是) = {:.2f}",
        "phrase": "要找的目标", "visible": "目标在画面里吗？", "where": "目标在哪？", "box": "框 {}", "nobox": "没找到",
        "crop": "放大复核", "tiles": "切块搜索", "tiles_v": "{} 个候选通过复核",
        "mode": "当前行为", "goal": "当前目标点", "hyps": "目标假设", "hyps_v": "{} 个（{} 个已确认，{} 个已排除）",
        "next": "下一个动作", "did": "刚执行", "none": "无",
        "desc": "画面描述", "reason": "推理", "plan": "计划的动作", "doing": "执行进度", "feedback": "上一个动作",
        "done_all": "全部执行完", "ok": "成功", "rejected": "被拒绝（撞到障碍）", "invalid": "输出无法解析，改为随机动作",
        "thinking": "正在问模型", "look": "正在看画面", "think_eb": "根据画面、指令和动作历史生成 JSON 动作计划",
        "success": "成功", "fail": "失败", "stopped": "已停止", "dist": "距目标 {:.2f} m",
        "ready": "正在启动模拟器……", "idle": "在右侧选择回合后点击“开始运行”",
        "legend": ["智能体", "当前目标点", "目标假设（越大证据越多）", "已排除", "已观测空地 / 障碍", "真实目标"],
        "modes": {"scan": "环视一周", "approach": "前往已确认的假设", "inspect": "靠近查看弱假设",
                  "explore": "探索没去过的地方", "push": "贴近目标", "": "开始"},
        "actions": ["前进 0.25 m", "后退 0.25 m", "右移 0.25 m", "左移 0.25 m", "右转 90°", "左转 90°", "抬头 30°",
                    "低头 30°"]},
    "en": {
        "title": "One 4B model, two ways to use it", "sep": ", ", "split": {"test": "test split", "dev": "dev split"},
        "subsets": {"base": "Base: category named", "common_sense": "Common-sense: described by use",
                    "complex_instruction": "Complex: long instruction", "visual_appearance": "Visual: appearance only",
                    "long_horizon": "Long-horizon: target behind"},
        "note_live": "Live: both agents query the same local model. The star is the true target, shown to the viewer only.",
        "note_replay": "Replay: actions, answers and decisions come from the saved test records. The star is the true "
                       "target, shown to the viewer only.",
        "s2m": "S2M (ours)", "eb": "EB prompt (benchmark interface)",
        "s2m_sub": "The VLM only answers whether and where the target is visible; geometry remembers and acts",
        "eb_sub": "The VLM looks at the image and writes a multi-step action plan (JSON)",
        "fpv": "agent's view", "map": "top-down map: S2M's own map and target hypotheses",
        "map_eb": "top-down view, for the viewer only (the EB prompt keeps no map)",
        "vlm": "the VLM's answers", "dec": "geometric controller", "vlm_eb": "the VLM's plan", "dec_eb": "execution",
        "step": "step {}", "of": " / {} max", "pyes": "P(yes) = {:.2f}",
        "phrase": "target phrase", "visible": "is the target visible?", "where": "where is it?", "box": "box {}",
        "nobox": "not found", "crop": "crop check", "tiles": "tiled search", "tiles_v": "{} candidates verified",
        "mode": "behavior", "goal": "current goal", "hyps": "hypotheses", "hyps_v": "{} ({} committed, {} ruled out)",
        "next": "next action", "did": "just did", "none": "none",
        "desc": "scene description", "reason": "reasoning", "plan": "planned actions", "doing": "progress",
        "feedback": "last action", "done_all": "all done", "ok": "succeeded", "rejected": "rejected (collision)",
        "invalid": "unparseable output, random action",
        "thinking": "asking the model", "look": "looking at the image",
        "think_eb": "writing a JSON action plan from the image, instruction and action history",
        "success": "SUCCESS", "fail": "FAILED", "stopped": "STOPPED", "dist": "{:.2f} m from target",
        "ready": "starting the simulator...", "idle": "pick an episode on the right and press Start",
        "legend": ["agent", "current goal", "hypothesis (size = evidence)", "ruled out", "observed free / obstacle",
                   "true target"],
        "modes": {"scan": "look around", "approach": "go to committed hypothesis", "inspect": "inspect weak hypothesis",
                  "explore": "explore unvisited space", "push": "close in", "": "start"},
        "actions": ["forward 0.25 m", "backward 0.25 m", "right 0.25 m", "left 0.25 m", "turn right 90°",
                    "turn left 90°", "look up 30°", "look down 30°"]},
}

_fonts = {}


def font(size, weight="Regular"):
    key = (size, weight)
    if key not in _fonts:
        path = os.path.join(FONT_DIR, f"NotoSansCJK-{weight}.ttc")
        _fonts[key] = ImageFont.truetype(path, size, index=2) if os.path.exists(path) else ImageFont.load_default()
    return _fonts[key]


def wrap(draw, text, fnt, width, max_lines):
    """Greedy wrap for mixed CJK/Latin text by measured width; a cut last line ends with an ellipsis."""
    lines, cur, text = [], "", str(text).replace("\n", " ")
    for i, ch in enumerate(text):
        if draw.textlength(cur + ch, font=fnt) <= width:
            cur += ch
            continue
        brk = cur.rfind(" ")
        if brk > len(cur) * 0.6 and ch != " ":
            lines.append(cur[:brk])
            cur = cur[brk + 1:] + ch
        else:
            lines.append(cur)
            cur = ch.lstrip()
        if len(lines) == max_lines:
            last = lines[-1]
            while last and draw.textlength(last + "…", font=fnt) > width:
                last = last[:-1]
            lines[-1] = last + "…"
            return lines
    if cur:
        lines.append(cur)
    return lines[:max_lines]


class MapView:
    """Fixed world-to-pixel mapping for one episode, shared by both panels (x to the right, z upwards)."""

    def __init__(self, floor_xz, extra_xz, size=IMG, margin=0.6):
        pts = np.asarray(list(floor_xz) + list(extra_xz), float)
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        self.c, self.size = (lo + hi) / 2, size
        self.s = (size - 16) / float((hi - lo).max())  # pixels per metre
        base = Image.new("RGB", (size, size), PANEL)
        d, h = ImageDraw.Draw(base), 0.05 * self.s
        for x, z in floor_xz:  # reachable floor: shows the room to the viewer; not an agent input
            u, v = self.px(x, z)
            d.rectangle([u - h, v - h, u + h, v + h], fill=FLOOR)
        self.base = base

    def px(self, x, z):
        return (self.size / 2 + (x - self.c[0]) * self.s, self.size / 2 - (z - self.c[1]) * self.s)

    def occupancy(self, occ_map):
        """S2M's own occupancy map over the floor: observed free floor and obstacle cells."""
        img = self.base.copy()
        if occ_map is None:
            return img
        v, u = np.mgrid[0:self.size, 0:self.size]
        i, j = occ_map.cell(self.c[0] + (u - self.size / 2) / self.s, self.c[1] - (v - self.size / 2) / self.s)
        ok = occ_map.inside(i, j)
        i, j = np.clip(i, 0, occ_map.n - 1), np.clip(j, 0, occ_map.n - 1)
        occ = ok & (occ_map.occ[i, j] >= 2) & ~occ_map.forced_free[i, j]
        free = ok & ((occ_map.free[i, j] > 0) | occ_map.forced_free[i, j]) & ~occ
        a = np.array(img)
        a[free], a[occ] = SEEN, OCC
        return Image.fromarray(a)


def _star(d, u, v, r, fill, outline=PANEL):
    d.polygon([(u + (r if k % 2 == 0 else r * .42) * math.sin(math.pi * k / 5),
                v - (r if k % 2 == 0 else r * .42) * math.cos(math.pi * k / 5)) for k in range(10)],
              fill=fill, outline=outline)


def _block(d, x, y, w, title, entries, color, bottom):
    """Titled label/value list; values wrap; stops before `bottom`."""
    d.rectangle([x, y + 3, x + 4, y + 21], fill=color)
    d.text((x + 12, y), title, font=font(18, "Bold"), fill=INK)
    y += 32
    for label, value, lines in entries:
        if y + 42 > bottom:
            break
        d.text((x, y), label, font=font(14), fill=INK2)
        y += 20
        for ln in wrap(d, value, font(16, "Medium"), w, lines):
            if y + 22 > bottom:
                break
            d.text((x, y), ln, font=font(16, "Medium"), fill=INK)
            y += 22
        y += 8
    return y


def _heading(yaw):
    y = math.radians(yaw)
    return math.sin(y), -math.cos(y)  # AI2-THOR yaw 0 = +z (up on the map), 90 = +x


class Panel:
    """One agent's 960x996 panel: set_key() redraws the static layer, frame() adds the dynamic layer."""

    def __init__(self, agent, lang, model, view, budget, target=None):
        self.agent, self.lang, self.model, self.view, self.budget = agent, lang, model, view, budget
        self.T, self.col, self.target = TEXT[lang], (BLUE if agent == "s2m" else ORANGE), target
        self.static = self.map_static = None

    def set_key(self, step, status, traj, vlm, dec, hyps=(), goal=None, occ_map=None, cur=-1):
        T, img = self.T, Image.new("RGB", (PW, PH), PAGE)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, PW, 64], fill=PANEL)
        d.rectangle([0, 0, 8, 64], fill=self.col)
        d.text((22, 6), T[self.agent], font=font(24, "Bold"), fill=INK)
        d.text((22, 38), T[self.agent + "_sub"], font=font(14), fill=INK2)
        s = T["step"].format(step) + T["of"].format(self.budget)
        d.text((PW - 20 - d.textlength(s, font=font(20, "Bold")), 18), s, font=font(20, "Bold"), fill=INK)
        d.text((FPV_X, IMG_Y - 22), T["fpv"], font=font(14, "Bold"), fill=INK2)
        d.text((MAP_X, IMG_Y - 22), T["map" if self.agent == "s2m" else "map_eb"], font=font(14, "Bold"), fill=INK2)
        cw = (MAP_X + IMG - FPV_X) / max(self.budget, 1)
        for k in range(self.budget):
            x0 = FPV_X + k * cw
            box = [x0 + 2, TL_Y, x0 + cw - 2, TL_Y + 22]
            st = status[k] if k < len(status) else "todo"
            if st == "ok":
                d.rounded_rectangle(box, radius=3, fill=self.col)
            elif st == "rej":
                d.rounded_rectangle(box, radius=3, fill=MUTED)
                d.text((x0 + cw / 2 - 5, TL_Y - 1), "×", font=font(16, "Bold"), fill=PANEL)
            else:
                d.rounded_rectangle(box, radius=3, outline=LINE, width=1)
            if k == cur:
                d.rounded_rectangle([x0, TL_Y - 2, x0 + cw, TL_Y + 24], radius=4, outline=INK, width=2)
        bottom = LEG_Y - 8
        _block(d, FPV_X, TX_Y, TX_W, T["vlm" if self.agent == "s2m" else "vlm_eb"], vlm, self.col, bottom)
        _block(d, MAP_X, TX_Y, TX_W, T["dec" if self.agent == "s2m" else "dec_eb"], dec, INK2, bottom)
        lx = FPV_X
        kinds = ["agent", "goal", "hyp", "ex", "map", "tgt"] if self.agent == "s2m" else ["agent", "tgt"]
        for kind, label in zip(["agent", "goal", "hyp", "ex", "map", "tgt"], T["legend"]):
            if kind not in kinds:
                continue
            y = LEG_Y
            if kind == "agent":
                d.ellipse([lx, y + 3, lx + 14, y + 17], fill=self.col)
            elif kind == "goal":
                d.ellipse([lx, y + 3, lx + 14, y + 17], outline=INK, width=2)
            elif kind == "hyp":
                d.ellipse([lx, y + 3, lx + 14, y + 17], fill=BLUE)
            elif kind == "ex":
                d.ellipse([lx, y + 3, lx + 14, y + 17], fill=PANEL, outline=BLUE, width=2)
            elif kind == "map":
                d.rectangle([lx, y + 4, lx + 7, y + 16], fill=SEEN)
                d.rectangle([lx + 7, y + 4, lx + 14, y + 16], fill=OCC)
            else:
                _star(d, lx + 7, y + 10, 8, INK)
            d.text((lx + 20, y), label, font=font(13), fill=INK2)
            lx += 20 + d.textlength(label, font=font(13)) + 18
        self.static = img
        m = self.view.occupancy(occ_map)
        md = ImageDraw.Draw(m)
        if len(traj) > 1:
            md.line([self.view.px(x, z) for x, z in traj], fill=self.col, width=3, joint="curve")
        for hx, hz, w, ex in hyps:
            u, v = self.view.px(hx, hz)
            r = 5 + 5 * min(max(w, 0.0), 2.5)
            if ex:
                md.ellipse([u - r, v - r, u + r, v + r], fill=PANEL, outline=BLUE, width=2)
                md.line([u - r * .6, v - r * .6, u + r * .6, v + r * .6], fill=BLUE, width=2)
            else:
                md.ellipse([u - r, v - r, u + r, v + r], fill=BLUE, outline=PANEL, width=2)
        if goal is not None:
            gu, gv = self.view.px(*goal)
            md.ellipse([gu - 7, gv - 7, gu + 7, gv + 7], outline=INK, width=2)
        if self.target is not None:
            _star(md, *self.view.px(*self.target), 11, INK)
        self.goal_px = None if goal is None else self.view.px(*goal)
        self.map_static = m

    def frame(self, rgb, pose, box=None, rejected=False, thinking=None, outcome=None):
        """Static layer + camera view + moving agent. thinking: (seconds, question) or None;
        outcome: (success, distance, aborted) or None."""
        T, img = self.T, self.static.copy()
        img.paste(Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).resize((IMG, IMG)), (FPV_X, IMG_Y))
        d = ImageDraw.Draw(img, "RGBA")
        k = IMG / rgb.shape[1]
        if box is not None:
            x0, y0, x1, y1 = (FPV_X + box[0] * k, IMG_Y + box[1] * k, FPV_X + box[2] * k, IMG_Y + box[3] * k)
            d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=PANEL, width=2)
            d.rectangle([x0, y0, x1, y1], outline=self.col, width=3)
        if rejected:
            d.rectangle([FPV_X, IMG_Y + IMG - 34, FPV_X + IMG, IMG_Y + IMG], fill=(0, 0, 0, 200))
            d.text((FPV_X + 12, IMG_Y + IMG - 31), "× " + T["rejected"], font=font(17, "Bold"), fill=PANEL)
        if thinking is not None:
            secs, question = thinking
            d.rectangle([FPV_X, IMG_Y + IMG - 62, FPV_X + IMG, IMG_Y + IMG], fill=(11, 11, 11, 190))
            dots = "●" * (1 + int(secs * 3) % 3) + "○" * (2 - int(secs * 3) % 3)
            d.text((FPV_X + 12, IMG_Y + IMG - 59), f"{dots}  {T['thinking']} · {secs:.1f} s", font=font(16, "Bold"),
                   fill=PANEL)
            for ln in wrap(d, question, font(15), IMG - 24, 1):
                d.text((FPV_X + 12, IMG_Y + IMG - 32), ln, font=font(15), fill=(230, 229, 226))
        if outcome is not None:
            ok, dist, aborted = outcome
            c = MUTED if aborted else (GOOD if ok else BAD)
            label = T["stopped"] if aborted else (("✓ " + T["success"]) if ok else ("× " + T["fail"]))
            bx, by = FPV_X + IMG / 2 - 140, IMG_Y + IMG / 2 - 44
            d.rounded_rectangle([bx, by, bx + 280, by + 88], radius=10, fill=PANEL + (245,), outline=c, width=3)
            d.text((bx + 20, by + 8), label, font=font(30, "Bold"), fill=c)
            if dist == dist:  # not NaN
                d.text((bx + 20, by + 54), T["dist"].format(dist), font=font(16, "Medium"), fill=INK)
        m = self.map_static.copy()
        md = ImageDraw.Draw(m, "RGBA")
        au, av = self.view.px(pose["x"], pose["z"])
        hu, hv = _heading(pose["yaw"])
        ang, r = math.degrees(math.atan2(hv, hu)), 2.2 * self.view.s
        md.pieslice([au - r, av - r, au + r, av + r], ang - 50, ang + 50, fill=self.col + (40,))
        if self.goal_px is not None and outcome is None:
            gu, gv = self.goal_px
            n = max(1, int(math.dist((au, av), (gu, gv)) / 7))
            for j in range(0, n, 2):
                md.line([(au + (gu - au) * j / n, av + (gv - av) * j / n),
                         (au + (gu - au) * min(j + 1, n) / n, av + (gv - av) * min(j + 1, n) / n)], fill=INK2, width=2)
        md.ellipse([au - 8, av - 8, au + 8, av + 8], fill=self.col, outline=PANEL, width=2)
        md.line([au, av, au + 15 * hu, av + 15 * hv], fill=INK, width=3)
        img.paste(m, (MAP_X, IMG_Y))
        d.rectangle([MAP_X, IMG_Y, MAP_X + IMG, IMG_Y + IMG], outline=LINE)
        return img


def placeholder(agent, lang, text):
    T, img = TEXT[lang], Image.new("RGB", (PW, PH), PAGE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, PW, 64], fill=PANEL)
    d.rectangle([0, 0, 8, 64], fill=BLUE if agent == "s2m" else ORANGE)
    d.text((22, 6), T[agent], font=font(24, "Bold"), fill=INK)
    d.text((22, 38), T[agent + "_sub"], font=font(14), fill=INK2)
    d.text(((PW - d.textlength(text, font=font(22, "Medium"))) / 2, PH / 2 - 20), text, font=font(22, "Medium"), fill=INK2)
    return img


def header(lang, mode, split=None, subset=None, episode=None, instruction=None):
    T, img = TEXT[lang], Image.new("RGB", (W, HEAD), PANEL)
    d = ImageDraw.Draw(img)
    d.text((24, 10), T["title"], font=font(26, "Bold"), fill=INK)
    if subset:
        d.text((24, 48), f"{T['split'][split]} · {T['subsets'][subset]} · #{episode}", font=font(16, "Medium"), fill=INK2)
    x = 24 + max(d.textlength(T["title"], font=font(26, "Bold")),
                 d.textlength(f"{T['split'].get(split, '')} · {T['subsets'].get(subset, '')} · #{episode}",
                              font=font(16, "Medium")) if subset else 0) + 40
    if instruction:
        for i, ln in enumerate(wrap(d, "“" + instruction + "”", font(20, "Bold"), W - x - 24, 1)):
            d.text((x, 12), ln, font=font(20, "Bold"), fill=INK)
    for ln in wrap(d, T["note_" + mode], font(14), W - x - 24, 1):
        d.text((x, 50), ln, font=font(14), fill=MUTED)
    d.line([0, HEAD - 1, W, HEAD - 1], fill=LINE)
    return img


def compose(head, left, right):
    img = Image.new("RGB", (W, H), PAGE)
    img.paste(head, (0, 0))
    img.paste(left, (0, HEAD))
    img.paste(right, (PW, HEAD))
    ImageDraw.Draw(img).line([PW, HEAD, PW, H], fill=LINE, width=2)
    return img


def title_card(lines, sub):
    img = Image.new("RGB", (W, H), PANEL)
    d, y = ImageDraw.Draw(img), H / 2 - 90
    for k, ln in enumerate(lines):
        f = font(44 if k == 0 else 30, "Bold" if k == 0 else "Medium")
        d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=INK)
        y += 66 if k == 0 else 48
    for ln in wrap(d, sub, font(20), W - 400, 3):
        d.text(((W - d.textlength(ln, font=font(20))) / 2, y + 20), ln, font=font(20), fill=INK2)
        y += 32
    return img


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class VideoWriter:
    """H.264 MP4 through ffmpeg (plays in browsers, PowerPoint, WeChat); OpenCV MPEG-4 as a fallback."""

    def __init__(self, path, size=(W, H), fps=25, preset="medium"):
        self.path, self.size, self.fps, self.frames = path, size, fps, 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        exe = ffmpeg_exe()
        self.proc = self.cv = None
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-crf", "21",
                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            import cv2
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, img, repeat=1):
        a = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
        for _ in range(repeat):
            if self.proc:
                self.proc.stdin.write(a.tobytes())
            else:
                self.cv.write(a[:, :, ::-1])
            self.frames += 1

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()
        return self.path
