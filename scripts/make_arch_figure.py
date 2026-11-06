"""Architecture figure of the paper (Fig. 1), in the CVPR-Figure house style, with an automatic audit.

  python scripts/make_arch_figure.py      (env embnav)
    -> paper/latex/fig/arch_*.png|pdf     real thumbnails from test episode long_horizon/42
    -> paper/latex/sec/fig_arch.tex       the figure (TikZ), written only if every check passes

House style (github.com/ChizkiyahuOhayon/CVPR-Figure, references/house-style.md): body serif (Times) at 7.5 pt,
stage titles 8.5 pt bold italic above grey dashed containers, captions 7 pt, annotations 6.5 pt italic; black
0.36 pt module outlines, 0.19 pt image hairlines, 0.6 pt arrows, 0.8 pt for the contribution; pale Office tints with
one meaning each (blue: frozen VLM, orange: S2M, green: actions, grey: inputs and inherited parts); text padding
7 / 4.2 pt, sibling gaps 16 / 9 pt, module radius 2.6 pt. The audit compiles the figure alone, reads every word
from the PDF, and fails on text outside its box, overlapping words or elements, arrows through elements, short
final arrow segments, off-palette colours, a fifth line weight, or a symbol that the caption does not define.
"""
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LATEX = os.path.join(ROOT, "paper", "latex")
FIG = os.path.join(LATEX, "fig")
REPLAY = os.path.join(ROOT, "paper", "figures", "scene_replay", "s2m_qwen3_r1_long_42")
EPISODE = os.path.join(ROOT, "runs", "eb_nav", "test", "s2m_Qwen3-VL-4B-Instruct-AWQ_final_r1", "long_horizon",
                       "episode_42.json")
# Office tint ladders (house style): pale fills carry meaning; deep tints only on arrows and small marks
OFFICE = {"oBlueL": "DEEBF7", "oBlueD": "2E75B6", "oOrangeP": "FDF2E9", "oOrangeL": "FBE5D6", "oOrangeD": "C55A11",
          "oOrangeS": "ED7D31", "oGreenS": "C5E0B4", "oGreyP": "F2F2F2", "oGreyL": "E7E6E6", "oGreyM": "BFBFBF",
          "oGreyS": "A6A6A6", "oGreyK": "7F7F7F", "oGreyD": "595959"}
HEX = {k: "#" + v for k, v in OFFICE.items()}
MM, PT = 1 / 25.4, 25.4 / 72
PADH, PADV, GH, GV = 7 * PT, 4.2 * PT, 16 * PT, 9 * PT   # text padding and sibling gaps (mm)
L1, L2 = 9 * PT + 2 * PADV, 18 * PT + 2 * PADV           # one- and two-line box heights at 7.5/9 pt
W, H = 174.0, 77.0
SIZES = {"label": 7.5, "title": 8.5, "caption": 7.0, "annot": 6.5}

# ------------------------------------------------------------------------------------------------ thumbnails


def thumbnails():
    from embagent.geometry import OccupancyMap
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    man = json.load(open(os.path.join(REPLAY, "manifest.json")))
    ep = json.load(open(EPISODE))
    log = {e["step"]: e for e in ep["perception_log"]}
    frames = {f["step"]: f for f in man["frames"]}
    frame = lambda s: np.asarray(Image.open(os.path.join(REPLAY, frames[s]["rgb"])).convert("RGB"))

    def save(img, name, size_mm, box=None):
        fig = plt.figure(figsize=(size_mm * MM, size_mm * MM), dpi=600)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, interpolation="lanczos")
        if box is not None:
            x0, y0, x1, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=HEX["oOrangeD"], lw=1.0))
        ax.set_axis_off()
        fig.savefig(os.path.join(FIG, f"arch_{name}.png"), dpi=600)
        plt.close(fig)

    save(frame(6), "frame", 22, log[6]["box"])  # the alarm clock in view, boxed by the VLM (presence 0.91)
    save(frame(0), "ebframe", 11)

    # bird's-eye view: occupancy from the replayed depth along the logged poses, the path, every hypothesis
    traj = ep["trajectory"]
    occ = OccupancyMap((traj[0]["x"], traj[0]["z"]), half_extent=4.0, res=0.05)
    for f in frames.values():
        d = np.asarray(Image.open(os.path.join(REPLAY, f["depth_mm"]))).astype(np.float32) / 1000.0
        occ.integrate(d, f["recorded_pose"])
    rgb = np.ones((occ.n, occ.n, 3)) * np.array([0xF2, 0xF2, 0xF2]) / 255          # unknown: grey.pale
    rgb[occ.known_free()] = 1.0                                                     # free: white
    rgb[occ.blocked(min_hits=2, inflate=0.0)] = np.array([0xBF, 0xBF, 0xBF]) / 255  # obstacle: grey.mid
    seen = {}
    for e in ep["perception_log"]:
        for hx, hz, w, _ in e["hyps"]:
            k = next((q for q in seen if np.hypot(q[0] - hx, q[1] - hz) < 0.5), (hx, hz))
            seen[k] = w
    xs, zs = [p["x"] for p in traj], [p["z"] for p in traj]
    allx, allz = xs + [k[0] for k in seen], zs + [k[1] for k in seen]
    cxm, czm = (min(allx) + max(allx)) / 2, (min(allz) + max(allz)) / 2
    half = max(max(allx) - min(allx), max(allz) - min(allz)) / 2 + 0.4
    fig = plt.figure(figsize=(28 * MM, 28 * MM), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ext = [occ.origin[0], occ.origin[0] + occ.n * occ.res, occ.origin[1], occ.origin[1] + occ.n * occ.res]
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
    ax.plot(xs, zs, color=HEX["oGreyD"], lw=0.9, solid_capstyle="round", zorder=3)
    ax.annotate("", xy=(xs[-1], zs[-1]), xytext=(xs[-2], zs[-2]),
                arrowprops=dict(arrowstyle="-|>", color=HEX["oGreyD"], lw=0.9, mutation_scale=6), zorder=4)
    ax.scatter([xs[0]], [zs[0]], s=9, marker="s", color=HEX["oGreyD"], zorder=4, lw=0)
    for (hx, hz), w in seen.items():
        if w >= 0.8:
            ax.scatter([hx], [hz], s=42, color=HEX["oOrangeD"], ec="white", lw=0.8, zorder=6)
        else:
            ax.scatter([hx], [hz], s=18, marker="x", color=HEX["oGreyK"], lw=1.0, zorder=5)
    ax.set_xlim(cxm - half, cxm + half)
    ax.set_ylim(czm - half, czm + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(os.path.join(FIG, "arch_bev.png"), dpi=600)
    plt.close(fig)

    # evidence of every hypothesis over the episode, with the commit threshold (dashed)
    def track(xz):
        return [(e["step"], next((h[2] for h in e["hyps"] if np.hypot(h[0] - xz[0], h[1] - xz[1]) < 0.5), np.nan))
                for e in ep["perception_log"]]
    plt.rcParams.update({"font.size": SIZES["annot"]})
    fig = plt.figure(figsize=(27 * MM, 16.0 * MM))
    ax = fig.add_axes([0.2, 0.36, 0.76, 0.6])
    for xz, c in (((0.33, -1.68), HEX["oOrangeD"]), ((1.52, -1.65), HEX["oGreyS"]), ((1.82, -1.82), HEX["oGreyS"])):
        tr = [(s, w) for s, w in track(xz) if not np.isnan(w)]
        ax.plot([s for s, _ in tr], [w for _, w in tr], color=c, lw=1.0, marker="o", ms=1.5, zorder=3)
    ax.axhline(0.8, color=HEX["oGreyD"], lw=0.5, ls=(0, (3.2, 2.2)), zorder=1)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(-1.6, 8.6)
    ax.set_xticks([0, 6, 12])
    ax.set_yticks([0, 4, 8])
    ax.set_xlabel("step", labelpad=1.2)
    ax.set_ylabel("$w_k$", labelpad=1.0)
    ax.tick_params(length=1.6, pad=1.2, width=0.36, colors="black")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.36)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_paper_figures import check
    check(fig, "evidence plot")
    fig.savefig(os.path.join(FIG, "arch_evidence.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ layout (mm, y down)
EL, ARROWS = [], []


def add(kind, name, rect, **kw):
    assert kind in ("container", "box", "img", "zone"), kind
    EL.append(dict(kind=kind, name=name, rect=tuple(round(v, 3) for v in rect), **kw))


def box(name, x0, y0, w, lines, fill, text):
    add("box", name, (x0, y0, x0 + w, y0 + (L1 if lines == 1 else L2)), fill=fill, text=text)
    return EL[-1]["rect"]


def mid(r):
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


FROZEN = "\\\\[-0.2mm]{\\fontsize{6.5}{7}\\selectfont\\itshape frozen}"
# (a) the benchmark's agent: same input, same frozen VLM, same output vocabulary as (b); only the operator differs
add("zone", "t_a", (0, 0, 70, 3.6), text="\\textbf{(a)\\ EB prompt (benchmark agent)}", size=SIZES["title"],
    align="left")
add("img", "a_frame", (0, 3.9, 11, 14.9), path="fig/arch_ebframe.png")
ya = 9.4 - L2 / 2
a_prompt = box("a_prompt", 11 + GH, ya, 25.0, 2, "oGreyL", "instruction $\\ell$,\\\\examples, log $h_t$")
a_vlm = box("a_vlm", a_prompt[2] + GH, ya, 21.0, 2, "oBlueL", "VLM $f$" + FROZEN)
a_plan = box("a_plan", a_vlm[2] + GH, ya, 26.0, 2, "oGreyL", "text plan of\\\\action ids")
a_act = box("a_act", 160.0, ya, 14.0, 2, "oGreenS", "actions\\\\$u_{t:t+m}$")
add("zone", "a_note", (a_plan[2] + 4.0, 5.2, a_act[0] - 4.0, 8.2), size=SIZES["annot"], align="center",
    italic=True, text="spatial state only in generated text")
add("zone", "a_fb", (118.0, 16.95, 134.0, 19.45), size=SIZES["caption"], align="center", fill="white",
    text="feedback")
ARROWS += [dict(pts=[(11, 9.4), (a_prompt[0], 9.4)]), dict(pts=[(a_prompt[2], 9.4), (a_vlm[0], 9.4)]),
           dict(pts=[(a_vlm[2], 9.4), (a_plan[0], 9.4)]), dict(pts=[(a_plan[2], 9.4), (a_act[0], 9.4)]),
           dict(pts=[(167.0, a_act[3]), (167.0, 18.2), (mid(a_prompt)[0], 18.2), (mid(a_prompt)[0], a_prompt[3])],
                under=["a_fb"])]

# (b) S2M: three stages between the same input frame and the same action output
add("zone", "t_b", (0, 21.0, 70, 24.6), text="\\textbf{(b)\\ \\ours (ours)}", size=SIZES["title"], align="left")
CY0, CY1 = 30.0, 72.0
S1 = (27.64, CY0, 27.64 + 21.0 + 2 * PADH, CY1)
S2 = (S1[2] + GH, CY0, S1[2] + GH + 27.0 + GH + 28.0 + 2 * PADH, CY1)
S3 = (S2[2] + GH, CY0, S2[2] + GH + 19.0 + 2 * PADH, CY1)
for name, rect, title, ours in (("S1", S1, "Per-image questions", False), ("S2", S2, "Metric hypothesis memory", True),
                                ("S3", S3, "Search and control", False)):
    add("container", name, rect, ours=ours)
    add("zone", f"t_{name}", (rect[0] + 0.6, 25.8, rect[2] + (4.6 if name == "S3" else 0), 29.4), text=title,
        size=SIZES["title"], align="left", bold=True, italic=True)
add("zone", "frame_lab", (0, 29.6, 22.0, 35.6), size=SIZES["caption"], align="center",
    text="frame $I_t$,\\\\phrase $s$")
add("img", "frame", (0, 36.4, 22.0, 58.4), path="fig/arch_frame.png")
# stage 1: the frozen VLM answers two questions about the current frame
vlm = box("vlm", S1[0] + PADH, 36.4, 21.0, 2, "oBlueL", "VLM $f$" + FROZEN)
ans = box("ans", S1[0] + PADH, vlm[3] + GV, 21.0, 2, "oGreyP", "presence $\\pi_t$\\\\box $b_t$")
# stage 2 (contribution): lift, fuse, and the resulting metric memory
xa = S2[0] + PADH
lift = box("lift", xa, CY0 + 2.0, 27.0, 1, "oOrangeL", "lift with $D_t$, $p_t$")
fuse = box("fuse", xa, lift[3] + GV, 27.0, 2, "oOrangeL", "fuse viewpoints,\\\\negative evidence")
add("img", "evidence", (xa, fuse[3] + GV, xa + 27.0, CY1 - 2.0), path="fig/arch_evidence.pdf", border=False)
xb = xa + 27.0 + GH
add("img", "bev", (xb, CY0 + 2.0, xb + 28.0, CY0 + 30.0), path="fig/arch_bev.png")
add("zone", "bev_lab", (xb, CY0 + 30.8, xb + 28.0, CY0 + 36.8), size=SIZES["caption"], align="center",
    text="hypotheses $\\mu_k$\\\\on the map $M_t$")
# stage 3: select a hypothesis and plan on the occupancy map
xc = S3[0] + PADH
sel = box("sel", xc, 41.0 - L1 / 2, 19.0, 1, "oOrangeL", "select $k^\\star$")
plan = box("plan", xc, sel[3] + GV, 19.0, 2, "oGreyL", "A* planner\\\\on $M_t$")
act = box("act", 160.0, mid(plan)[1] - L2 / 2, 14.0, 2, "oGreenS", "action\\\\$u_t$")
add("zone", "loop_lab", (76.0, 73.75, 104.0, 76.25), size=SIZES["caption"], align="center", fill="white",
    text="next frame")
yv, ya2, yl = mid(vlm)[1], mid(ans)[1], mid(lift)[1]
jog = (S1[2] + S2[0]) / 2
ARROWS += [
    dict(pts=[(22.0, yv), (vlm[0], yv)]),
    dict(pts=[(mid(vlm)[0], vlm[3]), (mid(vlm)[0], ans[1])], color="oBlueD"),
    dict(pts=[(ans[2], ya2), (jog, ya2), (jog, yl), (lift[0], yl)]),
    dict(pts=[(mid(lift)[0], lift[3]), (mid(lift)[0], fuse[1])], color="oOrangeD"),
    dict(pts=[(mid(fuse)[0], fuse[3]), (mid(fuse)[0], fuse[3] + GV)], color="oOrangeD"),
    dict(pts=[(fuse[2], mid(fuse)[1]), (xb, mid(fuse)[1])], color="oOrangeD"),
    dict(pts=[(xb + 28.0, mid(sel)[1]), (sel[0], mid(sel)[1])]),
    dict(pts=[(mid(sel)[0], sel[3]), (mid(sel)[0], plan[1])], color="oGreyD"),
    dict(pts=[(plan[2], mid(plan)[1]), (act[0], mid(plan)[1])]),
    dict(pts=[(167.0, act[3]), (167.0, 75.0), (11.0, 75.0), (11.0, 58.4)], under=["loop_lab"]),
]

# ------------------------------------------------------------------------------------------------ TikZ


def p(x, y):
    return f"({x:.2f},{-y:.2f})"


def font(size, bold=False, italic=False):
    return (f"\\rmfamily\\fontsize{{{size}}}{{{size * 1.2:.2f}}}\\selectfont" + ("\\bfseries" if bold else "")
            + ("\\itshape" if italic else ""))


def tikz():
    out = ["\\begin{tikzpicture}[x=1mm, y=1mm, >={Stealth[length=1.4mm, width=1.3mm]}]",
           f"\\useasboundingbox (0,0) rectangle {p(W, H)};"]
    for e in sorted(EL, key=lambda e: e["kind"] != "container"):
        x0, y0, x1, y1 = e["rect"]
        cx, cy = mid(e["rect"])
        if e["kind"] == "container":
            st = ("fill=oOrangeP, draw=oOrangeD, line width=0.8pt" if e["ours"] else "draw=oGreyK, line width=0.36pt")
            out.append(f"\\draw[{st}, dash pattern=on 3.2pt off 2.2pt, rounded corners=5pt] "
                       f"{p(x0, y0)} rectangle {p(x1, y1)};")
        elif e["kind"] == "box":
            out.append(f"\\filldraw[fill={e['fill']}, draw=black, line width=0.36pt, rounded corners=2.6pt] "
                       f"{p(x0, y0)} rectangle {p(x1, y1)};")
            out.append(f"\\node[font={font(SIZES['label'])}, text width={x1 - x0 - 2 * PADH:.2f}mm, align=center, "
                       f"inner sep=0, anchor=center] at {p(cx, cy)} {{{e['text']}}};")
        elif e["kind"] == "img":
            out.append(f"\\node[inner sep=0, anchor=north west] at {p(x0, y0)} "
                       f"{{\\includegraphics[width={x1 - x0:.2f}mm, height={y1 - y0:.2f}mm]{{{e['path']}}}}};")
            if e.get("border", True):
                out.append(f"\\draw[black, line width=0.19pt] {p(x0, y0)} rectangle {p(x1, y1)};")
        elif e["kind"] == "zone":
            if e.get("fill"):
                out.append(f"\\fill[{e['fill']}] {p(x0, y0)} rectangle {p(x1, y1)};")
            anchor, at = {"left": ("west", p(x0, cy)), "center": ("center", p(cx, cy))}[e["align"]]
            f = font(e["size"], e.get("bold", False), e.get("italic", False))
            out.append(f"\\node[font={f}, text width={x1 - x0:.2f}mm, align={e['align']}, inner sep=0, "
                       f"anchor={anchor}] at {at} {{{e['text']}}};")
    arrows = []
    for a in ARROWS:
        path = " -- ".join(p(x, y) for x, y in a["pts"])
        arrows.append(f"\\draw[->, line width=0.6pt, draw={a.get('color', 'black')}] {path};")
    n_cont = 2 + sum(1 for e in EL if e["kind"] == "container")  # containers are listed first in EL
    out[n_cont:n_cont] = arrows  # above the stage containers, below boxes and label patches
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


COLORS = "".join(f"\\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in OFFICE.items())
CAPTION = (
    "\\textbf{(a)}~The EB prompt maps $I_t$, the instruction $\\ell$, and the history $h_t$ to a text plan "
    "$u_{t:t+m}$ with the VLM $f$. \\textbf{(b)}~\\ours asks $f$ for the presence $\\pi_t$ and box $b_t$ of the "
    "phrase $s$, lifts them with depth $D_t$ and pose $p_t$ into hypotheses $\\mu_k$ with evidence $w_k$, selects "
    "$k^\\star$, and plans $u_t$ on the map $M_t$. Thumbnails: a Long-horizon test episode; the target (orange) passes "
    "the commit threshold (dashed) and two look-alikes (grey) fall below zero.")

# ------------------------------------------------------------------------------------------------ audit


def inside(r, x, y, m=0.0):
    return r[0] + m <= x <= r[2] - m and r[1] + m <= y <= r[3] - m


def raster_check(pdf, elements, arrows, name, dpi=600):
    """Render the compiled figure and check that what should be visible is: the fill of every box at an interior
    point clear of its text, and the colour of every arrow at the middle of each of its segments."""
    out = pdf[:-4] + "_ras"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, out], check=True)
    img = np.asarray(Image.open(out + ".png").convert("RGB")).astype(int)
    px = lambda x, y: img[min(img.shape[0] - 1, int(round(y / 25.4 * dpi))), min(img.shape[1] - 1, int(round(x / 25.4 * dpi)))]
    rgb = lambda c: np.array([255, 255, 255]) if c == "white" else (
        np.array([0, 0, 0]) if c == "black" else np.array([int(OFFICE[c][i:i + 2], 16) for i in (0, 2, 4)]))
    bad = []
    for e in elements:
        if e["kind"] not in ("box", "dia") or e.get("fill") is None:
            continue
        x0, y0, x1, y1 = e["rect"]
        # boxes: just inside the left edge; diamonds: near the left tip, both clear of the centred text
        x, y = (x0 + 0.9, (y0 + y1) / 2) if e["kind"] == "box" else (x0 + 0.12 * (x1 - x0), (y0 + y1) / 2)
        got = px(x, y)
        if np.abs(got - rgb(e["fill"])).max() > 14:
            bad.append(f"{name}: {e['name']} is covered (fill {e['fill']}, pixel {tuple(got)})")
    for a in arrows:
        col = rgb(a.get("color", "black"))
        for (xa, ya), (xb, yb) in zip(a["pts"], a["pts"][1:]):
            if np.hypot(xb - xa, yb - ya) < 2.0:
                continue
            x, y = (xa + xb) / 2, (ya + yb) / 2
            if any(e["name"] in a.get("under", []) and e["rect"][0] <= x <= e["rect"][2] and e["rect"][1] <= y <= e["rect"][3]
                   for e in elements):
                continue
            got = px(x, y)
            if np.abs(got - col).max() > 40:
                bad.append(f"{name}: arrow at ({x:.1f}, {y:.1f}) is hidden (colour {a.get('color', 'black')}, pixel {tuple(got)})")
                break
    return bad


def audit(body):
    bad = []
    # house-style invariants of the source
    used = set(re.findall(r"o(?:Blue|Orange|Green|Grey)[A-Z]", body)) | {a.get("color", "black") for a in ARROWS}
    off = [c for c in used if c not in OFFICE and c != "black"]
    if off:
        bad.append(f"off-palette colours: {off}")
    widths = set(re.findall(r"line width=([\d.]+)pt", body))
    if len(widths) > 4:
        bad.append(f"stroke-zoo: {sorted(widths)}")
    sizes = [float(s) for s in re.findall(r"\\fontsize\{([\d.]+)\}", body)]
    if min(sizes) < 6.5:
        bad.append(f"text-small: {min(sizes)} pt")
    hues = {a.get("color") for a in ARROWS if a.get("color", "black") not in ("black", "oGreyD")}
    if len(hues) > 3:
        bad.append(f"arrow-rainbow: {hues}")
    symbols = set(m for e in EL if e.get("text") for m in re.findall(r"\$[^$]+\$", e["text"]))
    missing = [s for s in symbols if s not in CAPTION]
    if missing:
        bad.append(f"undefined-symbol (not in the caption): {missing}")
    for e in EL:  # labels are noun phrases of at most six words, never sentences
        if e["kind"] == "box":
            plain = re.sub(r"\$[^$]+\$|\\\\[a-z]+(\[[^]]*\])?|\{|\}", " ", e["text"])
            n = len([w for w in re.split(r"[\s,]+", plain) if w.isalpha() and w != "frozen"])
            if n > 6 or plain.strip().endswith("."):
                bad.append(f"label-verbose: {e['name']} ({n} words)")
    # trace one input to one output: the frame must reach the action through the arrows
    def host(pt):
        hs = [e["name"] for e in EL if e["kind"] in ("box", "img") and inside(e["rect"], pt[0], pt[1], -0.05)]
        return hs[0] if hs else None
    graph = {}
    for a in ARROWS:
        u, v = host(a["pts"][0]), host(a["pts"][-1])
        if u and v:
            graph.setdefault(u, set()).add(v)
    for src, dst in (("frame", "act"), ("a_frame", "a_act")):
        seen, todo = {src}, [src]
        while todo:
            for n in graph.get(todo.pop(), ()):
                if n not in seen:
                    seen.add(n); todo.append(n)
        if dst not in seen:
            bad.append(f"unconnected: no arrow path from {src} to {dst}")
    for a in ARROWS:  # bent arrows need a final run of at least 2.5 head lengths (~10 pt)
        if len(a["pts"]) > 2:
            (xa, ya), (xb, yb) = a["pts"][-2], a["pts"][-1]
            if np.hypot(xb - xa, yb - ya) < 10 * PT:
                bad.append(f"short-final-segment: arrow ending at {a['pts'][-1]}")
    # compile the figure alone and read every word back
    test = os.path.join("/tmp", "arch_test")
    os.makedirs(test, exist_ok=True)
    src = os.path.join(test, "arch_test.tex")
    open(src, "w").write(
        "\\documentclass[border=0pt]{standalone}\n\\usepackage{fontspec}\n"
        "\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,"
        "BoldItalicFont=*-bolditalic]\n\\usepackage{amsmath,amssymb,graphicx,tikz}\n\\usetikzlibrary{arrows.meta}\n"
        f"\\graphicspath{{{{{LATEX}/}}}}\n\\newcommand{{\\ours}}{{S2M}}\n{COLORS}\n"
        f"\\begin{{document}}\n{body}\n\\end{{document}}\n")
    r = subprocess.run([os.path.expanduser("~/miniforge3/envs/tex/bin/tectonic"), "-X", "compile", src,
                        "--outdir", test], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("arch figure does not compile:\n" + r.stderr[-1500:])
    html = subprocess.run(["pdftotext", "-bbox", os.path.join(test, "arch_test.pdf"), "-"],
                          capture_output=True, text=True).stdout
    pw = float(re.search(r'<page width="([\d.]+)"', html).group(1)) * PT
    assert abs(pw - W) < 0.3, f"page width {pw:.1f} mm != {W}"
    words = [(float(a) * PT, float(b) * PT, float(c) * PT, float(d) * PT, t) for a, b, c, d, t in
             re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', html)]
    leaves = [e for e in EL if e["kind"] != "container"]
    for i, a in enumerate(leaves):
        for b in leaves[i + 1:]:
            ra, rb = a["rect"], b["rect"]
            if min(ra[2], rb[2]) - max(ra[0], rb[0]) > 0.05 and min(ra[3], rb[3]) - max(ra[1], rb[1]) > 0.05:
                bad.append(f"node-overlap: {a['name']} / {b['name']}")
        r0 = a["rect"]
        if r0[0] < 0 or r0[1] < 0 or r0[2] > W or r0[3] > H:
            bad.append(f"off-canvas: {a['name']}")
    for e in leaves:
        cx, cy = mid(e["rect"])
        for c in (c for c in EL if c["kind"] == "container" and inside(c["rect"], cx, cy)):
            r, q = c["rect"], e["rect"]
            if not (q[0] >= r[0] + 1.5 and q[2] <= r[2] - 1.5 and q[1] >= r[1] + 1.5 and q[3] <= r[3] - 1.5):
                bad.append(f"{e['name']} too close to the edge of {c['name']}")
    for x0, y0, x1, y1, t in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hosts = [e for e in leaves if inside(e["rect"], cx, cy)]
        if not hosts:
            bad.append(f"text-off-node: {t!r} at ({cx:.1f}, {cy:.1f})")
            continue
        e = hosts[0]
        if e["kind"] == "img":
            continue  # tick labels inside the evidence plot (checked by matplotlib)
        r = e["rect"]
        mh, mv = (PADH * 0.6, PADV * 0.35) if e["kind"] == "box" else (0.0, -0.3)
        if not (x0 >= r[0] + mh - 0.05 and x1 <= r[2] - mh + 0.05 and y0 >= r[1] + mv and y1 <= r[3] - mv):
            bad.append(f"label-overflow: {t!r} in {e['name']}")
    multi = [w for w in words if len(w[4]) > 1]  # single glyphs: accents and scripts overlap by design
    for i, a in enumerate(multi):
        for b in multi[i + 1:]:
            if min(a[2], b[2]) - max(a[0], b[0]) > 0.3 and min(a[3], b[3]) - max(a[1], b[1]) > 0.3:
                bad.append(f"label-overlap: {a[4]!r} / {b[4]!r}")
    for a in ARROWS:
        for (xa, ya), (xb, yb) in zip(a["pts"], a["pts"][1:]):
            if abs(xa - xb) > 1e-6 and abs(ya - yb) > 1e-6:
                bad.append(f"diagonal-edge at {(xa, ya)}")
            for t in np.linspace(0, 1, max(2, int(np.hypot(xb - xa, yb - ya) / 0.2)))[1:-1]:
                x, y = xa + t * (xb - xa), ya + t * (yb - ya)
                hit = [e["name"] for e in leaves if e["name"] not in a.get("under", []) and inside(e["rect"], x, y, 0.3)]
                if hit:
                    bad.append(f"edge-crosses-node: arrow {a['pts'][0]} -> {a['pts'][-1]} through {hit[0]}")
                    break
    bad += raster_check(os.path.join(test, "arch_test.pdf"), EL, ARROWS, "arch")
    bad = sorted(set(bad))
    if bad:
        raise SystemExit("arch figure audit failed:\n  " + "\n  ".join(bad))
    print(f"arch figure: audit ok ({len(words)} words, {len(leaves)} elements, {len(ARROWS)} arrows, "
          f"line weights {sorted(widths)}, min text {min(sizes)} pt, caption {len(CAPTION.split())} words)")


def main():
    thumbnails()
    body = tikz()
    audit(body)
    tex = ("\\begin{figure*}[t]\n\\centering\n"
           f"{COLORS}\n{body}\n\\caption{{{CAPTION}}}\n\\label{{fig:arch}}\n\\end{{figure*}}\n")
    open(os.path.join(LATEX, "sec", "fig_arch.tex"), "w").write(tex)
    print("wrote sec/fig_arch.tex")


if __name__ == "__main__":
    main()
