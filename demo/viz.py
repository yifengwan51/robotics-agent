"""Frame renderer and video writer for the S2M / EB-prompt demo videos.

One 1280x720 frame per moment of an episode:
  header   agent, model, instruction, step
  left     the agent's first-person RGB view, with the VLM's box (S2M) or the outcome of the last action
  right    top-down map: reachable floor (for the viewer), what S2M has observed, its trajectory, target
           hypotheses (sized by evidence), the current goal, its camera field of view, and the true target
  bottom   what the VLM was asked and answered, what the controller decided, and a step timeline
"""
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
# colors: series 1 / 2 of the paper figures; neutrals for text, floor, and map
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)
INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (150, 149, 144)
PANEL, PAGE = (255, 255, 255), (244, 243, 240)
FLOOR, SEEN, OCC = (229, 228, 224), (205, 222, 243), (96, 95, 92)
GOOD, BAD = (0, 131, 0), (227, 73, 72)
FONT_DIR = "/usr/share/fonts/opentype/noto"

TEXT = {
    "zh": {"s2m": "S2M（本文方法）", "eb": "EB prompt（基准自带接口）", "step": "第 {} / {} 步",
           "fpv": "智能体第一视角（RGB）", "map": "俯视地图", "map_eb": "俯视图（仅供观众参考，EB prompt 不维护地图）",
           "vlm": "小模型只回答两个问题", "dec": "几何控制器的决策", "vlm_eb": "模型输出的 JSON 计划",
           "dec_eb": "执行情况", "visible": "目标可见吗？", "where": "目标在哪？", "box": "画面中的框",
           "nobox": "没有框", "crop": "放大复核", "tiles": "切块搜索", "tiles_v": "{} 个候选通过复核",
           "phrase": "目标短语", "mode": "当前行为", "goal": "当前目标点", "hyps": "目标假设",
           "hyps_v": "{} 个（{} 个已确认、{} 个已排除）", "action": "下一个动作", "none": "无",
           "desc": "视觉描述", "reason": "推理", "plan": "计划动作", "feedback": "上一步反馈",
           "ok": "执行成功", "rejected": "被模拟器拒绝（碰撞）", "success": "成功", "fail": "失败",
           "dist": "距目标 {:.2f} m", "legend": ["智能体", "当前目标点", "目标假设（大小=证据）", "已排除假设",
                                                "真实目标（智能体不可见）", "已观测空地 / 障碍"],
           "modes": {"scan": "环视一周", "approach": "前往已确认的假设", "inspect": "靠近查看弱假设",
                     "explore": "探索未去过的区域", "push": "贴近目标", "": "开始"},
           "actions": ["前进 0.25 m", "后退 0.25 m", "右移 0.25 m", "左移 0.25 m", "右转 90°", "左转 90°",
                       "抬头 30°", "低头 30°"]},
    "en": {"s2m": "S2M (ours)", "eb": "EB prompt (benchmark interface)", "step": "step {} / {}",
           "fpv": "agent's first-person view (RGB)", "map": "top-down map",
           "map_eb": "top-down view (for the viewer only; the EB prompt keeps no map)",
           "vlm": "the small VLM answers two questions", "dec": "geometric controller", "vlm_eb": "model's JSON plan",
           "dec_eb": "execution", "visible": "is the target visible?", "where": "where is it?", "box": "box in frame",
           "nobox": "no box", "crop": "crop check", "tiles": "tiled search", "tiles_v": "{} candidates verified",
           "phrase": "target phrase", "mode": "behavior", "goal": "current goal", "hyps": "hypotheses",
           "hyps_v": "{} ({} committed, {} ruled out)", "action": "next action", "none": "none",
           "desc": "visual description", "reason": "reasoning", "plan": "planned actions",
           "feedback": "last action", "ok": "succeeded", "rejected": "rejected by the simulator (collision)",
           "success": "SUCCESS", "fail": "FAILED", "dist": "{:.2f} m from target",
           "legend": ["agent", "current goal", "hypothesis (size = evidence)", "ruled out",
                      "true target (hidden from agent)", "observed free / obstacle"],
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
    """Greedy wrap for mixed CJK/Latin text by measured width; the last line ends with an ellipsis if cut."""
    lines, cur = [], ""
    for ch in str(text).replace("\n", " "):
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
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    elif len(lines) == max_lines and (cur or len(text) > sum(map(len, lines))):
        last = lines[-1]
        while last and draw.textlength(last + "…", font=fnt) > width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines[:max_lines]


# layout (pixels): header, two square panels (first-person view, map), a text column, timeline, legend
HEAD, PAN, GAP = 64, 464, 16
FPV_X, MAP_X, COL_X, PAN_Y = GAP, GAP * 2 + PAN, GAP * 3 + PAN * 2, HEAD + 34
COL_W = W - COL_X - GAP
TL_Y, LEG_Y = PAN_Y + PAN + 22, PAN_Y + PAN + 64


class MapView:
    """Fixed world-to-pixel mapping for one episode's map panel (x to the right, z upwards)."""

    def __init__(self, floor_xz, extra_xz, size=PAN, margin=0.4):
        pts = np.asarray(list(floor_xz) + list(extra_xz), float)
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        self.c = (lo + hi) / 2
        self.s = (size - 16) / float((hi - lo).max())  # pixels per metre
        self.size = size
        self.floor = np.asarray(floor_xz, float)
        self._base = None

    def px(self, x, z):
        return (self.size / 2 + (x - self.c[0]) * self.s, self.size / 2 - (z - self.c[1]) * self.s)

    def base(self):
        """Reachable floor as light cells (cached): shows the room to the viewer; not an agent input."""
        if self._base is None:
            img = Image.new("RGB", (self.size, self.size), PANEL)
            d = ImageDraw.Draw(img)
            h = 0.05 * self.s
            for x, z in self.floor:
                u, v = self.px(x, z)
                d.rectangle([u - h, v - h, u + h, v + h], fill=FLOOR)
            self._base = img
        return self._base.copy()

    def with_occupancy(self, occ_map):
        """S2M's own occupancy map on top of the floor: observed free floor and obstacle cells."""
        img = self.base()
        if occ_map is None:
            return img
        v, u = np.mgrid[0:self.size, 0:self.size]
        x = self.c[0] + (u - self.size / 2) / self.s
        z = self.c[1] - (v - self.size / 2) / self.s
        i, j = occ_map.cell(x, z)
        ok = occ_map.inside(i, j)
        i, j = np.clip(i, 0, occ_map.n - 1), np.clip(j, 0, occ_map.n - 1)
        occ = ok & (occ_map.occ[i, j] >= 2) & ~occ_map.forced_free[i, j]
        free = ok & ((occ_map.free[i, j] > 0) | occ_map.forced_free[i, j]) & ~occ
        a = np.array(img)
        a[free] = SEEN
        a[occ] = OCC
        return Image.fromarray(a)


def _star(d, u, v, r, fill, outline=PANEL):
    pts = [(u + (r if k % 2 == 0 else r * 0.42) * math.sin(math.pi * k / 5),
            v - (r if k % 2 == 0 else r * 0.42) * math.cos(math.pi * k / 5)) for k in range(10)]
    d.polygon(pts, fill=fill, outline=outline)


def _dashed(d, a, b, fill, width=2, dash=7):
    n = max(1, int(math.dist(a, b) / dash))
    for k in range(0, n, 2):
        p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        q = (a[0] + (b[0] - a[0]) * min(k + 1, n) / n, a[1] + (b[1] - a[1]) * min(k + 1, n) / n)
        d.line([p, q], fill=fill, width=width)


def _heading_px(yaw):
    """AI2-THOR yaw (degrees, 0 = +z, 90 = +x) as a unit vector in map pixels (v grows downwards)."""
    y = math.radians(yaw)
    return math.sin(y), -math.cos(y)


def _block(d, x, y, w, title, entries, color):
    """A titled list of label / value entries; returns the y below it."""
    d.rectangle([x, y + 4, x + 4, y + 22], fill=color)
    d.text((x + 12, y), title, font=font(17, "Bold"), fill=INK)
    y += 34
    for label, value, lines in entries:
        d.text((x, y), label, font=font(13), fill=INK2)
        y += 19
        for ln in wrap(d, value, font(15, "Medium"), w, lines):
            d.text((x, y), ln, font=font(15, "Medium"), fill=INK)
            y += 21
        y += 7
    return y + 6


def render(view, st):
    """One frame. st: dict with agent ('s2m'/'eb'), lang, model, instruction, step, n_steps, rgb, box,
    occ_map, traj [(x,z)], pose, hyps [(x,z,w,exhausted)], goal (x,z)|None, target (x,z)|None,
    vlm / dec [(label, value, max_lines)], timeline ['ok'|'rej'|'todo'], outcome (success, dist)|None,
    rejected (bool), note."""
    T = TEXT[st["lang"]]
    col = BLUE if st["agent"] == "s2m" else ORANGE
    img = Image.new("RGB", (W, H), PAGE)
    d = ImageDraw.Draw(img)
    # header
    d.rectangle([0, 0, W, HEAD], fill=PANEL)
    d.rectangle([0, 0, 8, HEAD], fill=col)
    d.text((24, 8), T[st["agent"]], font=font(22, "Bold"), fill=INK)
    d.text((24, 38), st["model"], font=font(14), fill=INK2)
    stp = T["step"].format(st["step"], st["n_steps"])
    d.text((W - 24 - d.textlength(stp, font=font(20, "Bold")), 18), stp, font=font(20, "Bold"), fill=INK)
    ix = 24 + max(d.textlength(T[st["agent"]], font=font(22, "Bold")), d.textlength(st["model"], font=font(14))) + 36
    for k, ln in enumerate(wrap(d, "“" + st["instruction"] + "”", font(17, "Medium"), W - ix - 200, 2)):
        d.text((ix, 10 + 24 * k), ln, font=font(17, "Medium"), fill=INK)
    # first-person view
    d.text((FPV_X, PAN_Y - 26), T["fpv"], font=font(15, "Bold"), fill=INK2)
    fpv = Image.fromarray(st["rgb"]).resize((PAN, PAN))
    img.paste(fpv, (FPV_X, PAN_Y))
    if st.get("box") is not None:
        k = PAN / st["rgb"].shape[1]
        x0, y0, x1, y1 = [FPV_X + c * k if i % 2 == 0 else PAN_Y + c * k for i, c in enumerate(st["box"])]
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=PANEL, width=2)
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
    if st.get("rejected"):
        d.rectangle([FPV_X, PAN_Y + PAN - 34, FPV_X + PAN, PAN_Y + PAN], fill=(0, 0, 0))
        d.text((FPV_X + 12, PAN_Y + PAN - 30), "✕ " + T["rejected"], font=font(16, "Bold"), fill=PANEL)
    if st.get("outcome") is not None:
        ok, dist = st["outcome"]
        c = GOOD if ok else BAD
        bx, by = FPV_X + PAN / 2 - 150, PAN_Y + PAN / 2 - 46
        d.rounded_rectangle([bx, by, bx + 300, by + 92], radius=10, fill=PANEL, outline=c, width=3)
        d.text((bx + 22, by + 10), ("✓ " if ok else "✕ ") + T["success" if ok else "fail"], font=font(30, "Bold"), fill=c)
        d.text((bx + 22, by + 56), T["dist"].format(dist), font=font(16, "Medium"), fill=INK)
    # map
    d.text((MAP_X, PAN_Y - 26), T["map" if st["agent"] == "s2m" else "map_eb"], font=font(15, "Bold"), fill=INK2)
    m = view.with_occupancy(st.get("occ_map"))
    md = ImageDraw.Draw(m, "RGBA")
    pose = st["pose"]
    au, av = view.px(pose["x"], pose["z"])
    hu, hv = _heading_px(pose["yaw"])
    ang = math.degrees(math.atan2(hv, hu))
    r = 2.2 * view.s
    md.pieslice([au - r, av - r, au + r, av + r], ang - 50, ang + 50, fill=col + (40,))
    if len(st["traj"]) > 1:
        md.line([view.px(x, z) for x, z in st["traj"]], fill=col, width=3, joint="curve")
    for hx, hz, w, ex in st.get("hyps", []):
        u, v = view.px(hx, hz)
        rr = 5 + 5 * min(max(w, 0.0), 2.5)
        if ex:
            md.ellipse([u - rr, v - rr, u + rr, v + rr], fill=PANEL, outline=col, width=2)
            md.line([u - rr * .6, v - rr * .6, u + rr * .6, v + rr * .6], fill=col, width=2)
        else:
            md.ellipse([u - rr, v - rr, u + rr, v + rr], fill=col, outline=PANEL, width=2)
    if st.get("goal") is not None:
        gu, gv = view.px(*st["goal"])
        _dashed(md, (au, av), (gu, gv), INK2)
        md.ellipse([gu - 6, gv - 6, gu + 6, gv + 6], outline=INK, width=2)
    if st.get("target") is not None:
        _star(md, *view.px(*st["target"]), 11, INK)
    md.ellipse([au - 8, av - 8, au + 8, av + 8], fill=col, outline=PANEL, width=2)
    md.line([au, av, au + 15 * hu, av + 15 * hv], fill=INK, width=3)
    img.paste(m, (MAP_X, PAN_Y))
    d.rectangle([MAP_X, PAN_Y, MAP_X + PAN, PAN_Y + PAN], outline=(214, 213, 208))
    # text column
    y = _block(d, COL_X, PAN_Y - 26, COL_W, T["vlm" if st["agent"] == "s2m" else "vlm_eb"], st["vlm"], col)
    _block(d, COL_X, y, COL_W, T["dec" if st["agent"] == "s2m" else "dec_eb"], st["dec"], INK2)
    # timeline
    n = len(st["timeline"])
    cw = (MAP_X + PAN - FPV_X) / max(n, 1)
    for k, s in enumerate(st["timeline"]):
        x0 = FPV_X + k * cw
        box = [x0 + 2, TL_Y, x0 + cw - 2, TL_Y + 22]
        if s == "ok":
            d.rounded_rectangle(box, radius=3, fill=col)
        elif s == "rej":
            d.rounded_rectangle(box, radius=3, fill=MUTED)
            d.text((x0 + cw / 2 - 5, TL_Y), "✕", font=font(14, "Bold"), fill=PANEL)
        else:
            d.rounded_rectangle(box, radius=3, outline=MUTED, width=1)
        if k == st.get("cur", st["step"] - 1):
            d.rounded_rectangle([x0, TL_Y - 2, x0 + cw, TL_Y + 24], radius=4, outline=INK, width=2)
    # legend (map symbols)
    lx, ly = FPV_X, LEG_Y
    kinds = ["agent", "goal", "hyp", "ex", "tgt", "map"]
    shown = kinds if st["agent"] == "s2m" else ["agent", "tgt"]
    for kind, label in zip(kinds, T["legend"]):
        if kind not in shown:
            continue
        if kind == "agent":
            d.ellipse([lx, ly + 3, lx + 14, ly + 17], fill=col)
        elif kind == "goal":
            d.ellipse([lx, ly + 3, lx + 14, ly + 17], outline=INK, width=2)
        elif kind == "hyp":
            d.ellipse([lx, ly + 3, lx + 14, ly + 17], fill=BLUE)
        elif kind == "ex":
            d.ellipse([lx, ly + 3, lx + 14, ly + 17], fill=PANEL, outline=BLUE, width=2)
        elif kind == "tgt":
            _star(d, lx + 7, ly + 10, 8, INK)
        else:
            d.rectangle([lx, ly + 4, lx + 7, ly + 16], fill=SEEN)
            d.rectangle([lx + 7, ly + 4, lx + 14, ly + 16], fill=OCC)
        d.text((lx + 20, ly), label, font=font(13), fill=INK2)
        lx += 20 + d.textlength(label, font=font(13)) + 22
    if st.get("note"):
        for k, ln in enumerate(wrap(d, st["note"], font(12), COL_W, 3)):
            d.text((COL_X, H - 62 + 17 * k), ln, font=font(12), fill=MUTED)
    return img


def title_card(lines, sub, lang, size=(W, H)):
    """Plain intro frame: a two-line headline and a subtitle."""
    img = Image.new("RGB", size, PANEL)
    d = ImageDraw.Draw(img)
    y = size[1] / 2 - 70
    for k, ln in enumerate(lines):
        f = font(34 if k == 0 else 24, "Bold" if k == 0 else "Medium")
        d.text(((size[0] - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=INK)
        y += 52 if k == 0 else 38
    for ln in wrap(d, sub, font(17), size[0] - 240, 3):
        d.text(((size[0] - d.textlength(ln, font=font(17))) / 2, y + 16), ln, font=font(17), fill=INK2)
        y += 26
    return img


def side_by_side(a, b):
    img = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), PAGE)
    img.paste(a, (0, 0))
    img.paste(b, (a.width, 0))
    return img


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class VideoWriter:
    """MP4 writer: H.264 (plays in browsers, PowerPoint, WeChat) through ffmpeg when available, otherwise
    OpenCV's MPEG-4 Part 2 encoder."""

    def __init__(self, path, size, fps=10):
        self.path, self.size, self.fps = path, size, fps
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        exe = _ffmpeg()
        self.proc = self.cv = None
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            import cv2
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.frames = 0

    def write(self, img, repeat=1):
        a = np.asarray(img.convert("RGB"))
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
