"""Open-vocabulary target grounding with a local VLM (OpenAI-compatible server), and depth lifting.

Coordinate conventions differ by model family:
- Qwen2.5-VL: absolute pixels of the processor-resized image (sides rounded to multiples of 28).
- Qwen3-VL / Qwen3.5: relative coordinates in [0, 1000].
"""
import base64
import io
import json
import math
import os
import re

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

from embagent.geometry import pixel_to_world

BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")


def data_url(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def qwen25_resized(h, w, factor=28, min_pixels=3136, max_pixels=409600):
    """Qwen2.5-VL smart_resize (matches the --mm-processor-kwargs max_pixels used by scripts/serve_vlm.sh)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


class Grounder:
    def __init__(self, model, base_url=None, coord_mode=None, timeout=120):
        self.model = model
        self.client = OpenAI(base_url=base_url or BASE_URL, api_key="EMPTY", timeout=timeout)
        self.coord_mode = coord_mode or ("abs" if "2.5-VL" in model or "2_5" in model else "rel1000")
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if "3.5" in model else {}
        self.calls = 0

    def _chat(self, rgb, text, max_tokens=128, **kw):
        self.calls += 1
        content = [{"type": "image_url", "image_url": {"url": data_url(rgb)}}, {"type": "text", "text": text}] \
            if rgb is not None else [{"type": "text", "text": text}]
        return self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], temperature=0,
            max_tokens=max_tokens, extra_body=self.extra or None, **kw)

    def _to_pixels(self, box, h, w):
        x1, y1, x2, y2 = box
        if self.coord_mode == "rel1000":
            sx, sy = w / 1000.0, h / 1000.0
        else:
            rh, rw = qwen25_resized(h, w)
            sx, sy = w / rw, h / rh
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        return [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]

    def presence(self, rgb, phrase):
        """P(target visible) from the first-token yes/no logprobs."""
        r = self._chat(rgb, f"Does this image show {_clean(phrase)}? Answer yes or no.",
                       max_tokens=1, logprobs=True, top_logprobs=10)
        tops = r.choices[0].logprobs.content[0].top_logprobs if r.choices[0].logprobs else []
        py = sum(math.exp(t.logprob) for t in tops if t.token.strip().lower() == "yes")
        pn = sum(math.exp(t.logprob) for t in tops if t.token.strip().lower() == "no")
        return py / (py + pn) if py + pn > 0 else 0.5

    def ground(self, rgb, phrase):
        """Bounding box [x1, y1, x2, y2] in image pixels, or None."""
        h, w = rgb.shape[:2]
        phrase = _clean(phrase)
        r = self._chat(rgb, f'Locate the {phrase} in the image and output its bounding box as JSON: '
                            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{phrase}"}}]. '
                            f'If it is not in the image, output [].', max_tokens=96)
        text = r.choices[0].message.content or ""
        m = _BOX_RE.search(text)
        if not m:
            return None, text
        box = self._to_pixels([float(v) for v in m.groups()], h, w)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return None, text
        return box, text

    def target_phrase(self, instruction):
        """Rewrite an indirect instruction into a short visual description of the target object."""
        return self.target_spec(instruction)["query"]

    def target_spec(self, instruction):
        """{'name': common object name or '', 'looks': short appearance, 'query': phrase used for grounding}.

        Grounding with the plain category name is more reliable for small VLMs than long descriptions;
        appearance-only instructions fall back to the description."""
        r = self._chat(None, "A robot must navigate to one object in a house. Instruction: "
                             f'"{instruction}"\nReply with JSON {{"name": "...", "appearance_only": true or false, '
                             '"looks": "..."} where\n'
                             "- name: the most likely specific household object, as its common name in 1-2 words "
                             '(e.g. "toaster", "trash can", "laptop"; never a vague word like "device" or "object").\n'
                             "- appearance_only: true if the instruction describes the object only by colour, "
                             "shape, material or texture, without saying what it is or what it is used for.\n"
                             "- looks: the appearance words from the instruction (colour, shape, material), "
                             "empty if none.", max_tokens=96)
        text = (r.choices[0].message.content or "").strip()
        try:
            spec = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except Exception:
            spec = {"name": "", "appearance_only": True, "looks": text[:80]}
        name = _clean(spec.get("name", ""))
        looks = _clean(spec.get("looks", ""))
        named = bool(name) and not bool(spec.get("appearance_only", False))
        if not named and not looks:  # no appearance words extracted: fall back to the instruction itself
            looks = _clean(instruction)
        return {"named": named, "name": name, "looks": looks, "query": name if named else looks}

    def tiles(self, rgb, grid=3, overlap=0.2, out=448):
        """Overlapping magnified tiles for small-object search: list of (tile_rgb, x0, y0, tw, th)."""
        H, W = rgb.shape[:2]
        tw = int(W / (grid - (grid - 1) * overlap))
        th = int(H / (grid - (grid - 1) * overlap))
        res = []
        for y0 in np.linspace(0, H - th, grid).astype(int):
            for x0 in np.linspace(0, W - tw, grid).astype(int):
                tile = np.asarray(Image.fromarray(rgb[y0:y0 + th, x0:x0 + tw]).resize((out, out), Image.BICUBIC))
                res.append((tile, int(x0), int(y0), tw, th))
        return res

    def ground_tile(self, tile, phrase):
        """Ground in one tile; returns the box in full-image pixels or None."""
        img, x0, y0, tw, th = tile
        box, _ = self.ground(img, phrase)
        if box is None:
            return None
        sx, sy = tw / img.shape[1], th / img.shape[0]
        return [x0 + box[0] * sx, y0 + box[1] * sy, x0 + box[2] * sx, y0 + box[3] * sy]

    def ground_all(self, rgb, phrase, max_n=6):
        """Every box the VLM thinks could be the target (candidate set for comparison), in image pixels."""
        h, w = rgb.shape[:2]
        phrase = _clean(phrase)
        r = self._chat(rgb, f'Locate every object in the image that could be the {phrase}. Output JSON: '
                            f'[{{"bbox_2d": [x1, y1, x2, y2]}}, ...]. Output [] if there is none.', max_tokens=256)
        boxes = []
        for m in _BOX_RE.finditer(r.choices[0].message.content or ""):
            b = self._to_pixels([float(v) for v in m.groups()], h, w)
            if b[2] - b[0] >= 3 and b[3] - b[1] >= 3 and all(_iou(b, o) < 0.5 for o in boxes):
                boxes.append(b)
        return boxes[:max_n]

    def choose(self, rgb, boxes, phrase):
        """Set-of-Mark comparison: probability that each numbered box is the target, plus 'none' (index 0).

        A relative judgement among visible candidates, read from the first-token distribution over digits."""
        img = Image.fromarray(rgb).copy()
        draw = ImageDraw.Draw(img)
        colors = [(255, 40, 40), (40, 160, 255), (40, 220, 40), (255, 200, 0), (220, 60, 255), (0, 230, 230)]
        for k, b in enumerate(boxes):
            c = colors[k % len(colors)]
            draw.rectangle(b, outline=c, width=3)
            tx, ty = b[0], max(0, b[1] - 18)
            draw.rectangle([tx, ty, tx + 16, ty + 18], fill=c)
            draw.text((tx + 4, ty + 2), str(k + 1), fill=(0, 0, 0))
        n = len(boxes)
        r = self._chat(np.asarray(img), f'Numbered boxes mark candidate objects. Which one is the {_clean(phrase)}? '
                                        f'Answer with its number only (1-{n}), or 0 if none of them is.',
                       max_tokens=1, logprobs=True, top_logprobs=20)
        tops = r.choices[0].logprobs.content[0].top_logprobs if r.choices[0].logprobs else []
        p = np.zeros(n + 1)
        for t in tops:
            s = t.token.strip()
            if s.isascii() and s.isdigit() and int(s) <= n:
                p[int(s)] += math.exp(t.logprob)
        return p / p.sum() if p.sum() > 0 else np.full(n + 1, 1.0 / (n + 1))

    def zoom_presence(self, rgb, box, phrase, scale=2.5, out=448):
        """Presence probability on a magnified crop around `box` (verifies small or uncertain detections)."""
        H, W = rgb.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = max(32.0, scale * max(box[2] - box[0], box[3] - box[1]) / 2)
        x1, x2 = int(max(0, cx - half)), int(min(W, cx + half))
        y1, y2 = int(max(0, cy - half)), int(min(H, cy + half))
        crop = Image.fromarray(rgb[y1:y2, x1:x2]).resize((out, out), Image.BICUBIC)
        return self.presence(np.asarray(crop), phrase)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _clean(phrase):
    p = str(phrase or "").strip().strip(".").strip('"').lower()
    for art in ("the ", "a ", "an "):
        if p.startswith(art):
            p = p[len(art):]
    for tail in (" in the room", " in the house", " in the kitchen"):
        if p.endswith(tail):
            p = p[: -len(tail)]
    return p.strip()


def lift_box(box, depth, pose, fov=100.0, inner=0.5, max_depth=10.0, cam_h=None):
    """World (x, z, height, range) of a box via the median depth of its central region."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(1.0, (x2 - x1) * inner / 2), max(1.0, (y2 - y1) * inner / 2)
    H, W = depth.shape
    patch = depth[int(max(0, cy - hh)):int(min(H, cy + hh + 1)), int(max(0, cx - hw)):int(min(W, cx + hw + 1))]
    vals = patch[(patch > 0.05) & (patch < max_depth)]
    if vals.size == 0:
        return None
    d = float(np.median(vals))
    kw = {} if cam_h is None else {"cam_h": cam_h}
    x, z, hgt = pixel_to_world(cx, cy, d, pose, fov=fov, img_w=W, img_h=H, **kw)
    return float(x), float(z), float(hgt), d
