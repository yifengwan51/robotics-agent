"""Paper figure: what S2M does in one step, as a flowchart (TikZ, house style of Fig. 1), with an automatic audit.

  python scripts/make_flow_figure.py      (env embnav)  -> paper/latex/sec/fig_mech.tex

Every box and decision follows embagent/planners/s2m.py (perceive, _add, _negative_evidence, _select, act).
The text of every node is measured first, the layout is computed from the measured widths, and the figure is
written only if the audit passes: text inside its box or diamond, no overlapping nodes or words, no arrow through
a node, orthogonal arrows only, a path from the input to the action, and every symbol defined in the caption.
"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_arch_figure import COLORS, OFFICE, PADV, PT, SIZES, font, raster_check  # noqa: E402  (house style of Fig. 1)

LATEX = os.path.join(ROOT, "paper", "latex")
TECTONIC = os.path.expanduser("~/miniforge3/envs/tex/bin/tectonic")
W = 174.0
L1, L2, L3 = 9 * PT + 2 * PADV, 18 * PT + 2 * PADV, 27 * PT + 2 * PADV
MARGIN, GAP_V, GAP_H, GAP_C, PAD, DH = 1.9, 8 * PT, 4.6, 3.2, 1.6, 9.2
TOP = 4.4  # top of the stage containers; stage titles sit above

T = {  # node text; every symbol is defined in the caption
    "in": "$I_t$, $s$",
    "vlm": "VLM $f$\\\\$\\pi_t$, $b_t$",
    "weak": "weak?",
    "crop": "crop check,\\\\tiles",
    "weight": "weight $a_t$",
    "lift": "lift to $q_t$ ($D_t$, $p_t$)",
    "new": "\\textit{otherwise:}\\\\$\\mu_k=q_t$, $w_k=a_t$",
    "fuse": "\\textit{within $r$ of $\\mu_k$:}\\\\fuse, $w_k \\mathrel{+}= \\gamma_t a_t$",
    "neg": "\\textit{in view, undetected:}\\\\$w_k \\mathrel{-}= \\lambda$; drop at $w_k\\le-1$",
    "commit": "$w_k\\ge\\tau_c$?",
    "goto": "go to $\\mu_{k^\\star}$;\\\\exhaust it\\\\on arrival",
    "else": "look around,\\\\inspect,\\\\or explore",
    "astar": "A* on $M_t$",
    "out": "action\\\\$u_t$",
}
CAPTION = ("One step of \\ours. The frozen VLM $f$ gives the presence $\\pi_t$ and box $b_t$ of the phrase $s$ on the "
           "frame $I_t$; a weak answer is checked on a crop or on tiles, which sets its weight $a_t$. The box is lifted to "
           "a floor point $q_t$ with depth $D_t$ and pose $p_t$ and fused into a hypothesis $\\mu_k$ within $r$, or starts "
           "one; $\\gamma_t<1$ discounts a repeated view, and a hypothesis that should be seen but is not loses "
           "$\\lambda$ of its evidence $w_k$. Above $\\tau_c$ a hypothesis is committed, and the agent goes to "
           "$\\mu_{k^\\star}$; A* on the map $M_t$ gives the action $u_t$.")
EL, ARROWS = [], []


def compile_tex(body, name):
    d = os.path.join("/tmp", name)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, f"{name}.tex")
    open(src, "w").write(
        "\\documentclass[border=0pt]{standalone}\n\\usepackage{fontspec}\n"
        "\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,"
        "BoldItalicFont=*-bolditalic]\n\\usepackage{amsmath,amssymb,tikz}\n\\usetikzlibrary{arrows.meta}\n"
        f"\\newcommand{{\\ours}}{{S2M}}\n{COLORS}\n\\begin{{document}}\n{body}\n\\end{{document}}\n")
    r = subprocess.run([TECTONIC, "-X", "compile", src, "--outdir", d], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{name} does not compile:\n" + r.stderr[-1500:])
    html = subprocess.run(["pdftotext", "-bbox", os.path.join(d, f"{name}.pdf"), "-"],
                          capture_output=True, text=True).stdout
    pw = float(re.search(r'<page width="([\d.]+)"', html).group(1)) * PT
    words = [(float(a) * PT, float(b) * PT, float(c) * PT, float(e) * PT, t) for a, b, c, e, t in
             re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', html)]
    return pw, words


def measure():
    """Width (mm) of every node text at the label size, from the compiled glyphs."""
    keys = list(T)
    nodes = "".join(f"\\node[font={font(SIZES['label'])}, align=center, inner sep=0, anchor=north west] "
                    f"at (0,{-12 * i}mm) {{{T[k]}}};\n" for i, k in enumerate(keys))
    _, words = compile_tex("\\begin{tikzpicture}\n\\fill[white] (0,0) rectangle (1mm,1mm);\n" + nodes
                           + "\\end{tikzpicture}", "flow_measure")
    tw = {}
    for i, k in enumerate(keys):
        band = [w for w in words if i * 12 - 1 <= w[1] - 1.0 < (i + 1) * 12 - 1]
        tw[k] = max(w[2] for w in band) - min(w[0] for w in band)
    return tw


def box(name, cx, top, w, lines, fill):
    h = {1: L1, 2: L2, 3: L3}[lines]
    EL.append(dict(kind="box", name=name, rect=(cx - w / 2, top, cx + w / 2, top + h), fill=fill, text=T[name]))
    return EL[-1]["rect"]


def dia(name, cx, top, w):
    EL.append(dict(kind="dia", name=name, rect=(cx - w / 2, top, cx + w / 2, top + DH), fill="white", text=T[name]))
    return EL[-1]["rect"]


def zone(name, rect, text, size, align="center", italic=False, bold=False, fill=None):
    EL.append(dict(kind="zone", name=name, rect=rect, text=text, size=size, align=align, italic=italic, bold=bold,
                   fill=fill))


def cy(r):
    return (r[1] + r[3]) / 2


def cx(r):
    return (r[0] + r[2]) / 2


def arrow(pts, color="black", under=()):
    ARROWS.append(dict(pts=pts, color=color, under=list(under)))


def yes_no(d, side, text):
    """Branch label beside the wire leaving diamond d to the right ('r') or downwards ('b')."""
    if side == "r":
        zone(f"{text}_{len(EL)}", (d[2] + 0.5, cy(d) - 3.1, d[2] + GAP_H - 0.5, cy(d) - 0.5), text, SIZES["annot"],
             italic=True)
    else:
        zone(f"{text}_{len(EL)}", (cx(d) + 0.6, d[3] + 0.2, cx(d) + 5.0, d[3] + GAP_V - 0.2), text, SIZES["annot"],
             align="left", italic=True)


def layout(tw):
    EL.clear(); ARROWS.clear()
    bw = lambda k: tw[k] + 2 * MARGIN
    # rhombus of height DH whose inner diamond (inset 1 mm) holds a text box of width tw and half-height 1.7 mm
    dw = lambda k: 2 * (1.0 + (tw[k] / 2) / (1 - 1.7 / (DH / 2 - 1.0)))
    c0 = TOP + PAD
    # input
    w_in = bw("in")
    # stage 1: per-image questions
    c1l = w_in + GAP_C
    s1 = max(bw("vlm"), dw("weak"), bw("weight"))
    x1 = c1l + PAD + s1 / 2
    vlm = box("vlm", x1, c0, bw("vlm"), 2, "oBlueL")
    weak = dia("weak", x1, vlm[3] + GAP_V, dw("weak"))
    crop = box("crop", weak[2] + GAP_H + bw("crop") / 2, cy(weak) - L2 / 2, bw("crop"), 2, "oBlueL")
    weight = box("weight", x1, weak[3] + GAP_V, bw("weight"), 1, "oGreyP")
    c1r = max(crop[2], x1 + s1 / 2) + PAD
    EL.insert(0, dict(kind="box", name="in", rect=(0, cy(vlm) - L1 / 2, w_in, cy(vlm) + L1 / 2), fill="oGreyL",
                      text=T["in"]))
    # stage 2: metric hypothesis memory (the contribution): lift, then fuse or create, then negative evidence
    c2l = c1r + GAP_C
    wa = max(bw("lift"), bw("fuse"))
    xa = c2l + PAD + wa / 2
    lift = box("lift", xa, c0, bw("lift"), 1, "oOrangeL")
    fuse = box("fuse", xa, lift[3] + GAP_V, bw("fuse"), 2, "oOrangeL")
    xb = xa + wa / 2 + GAP_H + bw("new") / 2
    new = box("new", xb, fuse[1], bw("new"), 2, "oOrangeL")
    nl = xa - wa / 2
    wn = max(bw("neg"), new[2] - nl)
    neg = box("neg", nl + wn / 2, fuse[3] + GAP_V, wn, 2, "oOrangeL")
    c2r = max(new[2], neg[2]) + PAD
    # stage 3: search and control
    c3l = c2r + GAP_C
    s3 = max(dw("commit"), bw("else"))
    x3 = c3l + PAD + s3 / 2
    commit = dia("commit", x3, c0, dw("commit"))
    xr = commit[2] + GAP_H + max(bw("goto"), bw("astar")) / 2
    goto = box("goto", xr, c0, bw("goto"), 3, "oOrangeL")
    els = box("else", x3, commit[3] + GAP_V, bw("else"), 3, "oGreyL")
    astar = box("astar", xr, els[3] + GAP_V, bw("astar"), 1, "oGreyL")
    c3r = max(goto[2], astar[2], x3 + s3 / 2) + PAD
    out = (c3r + GAP_C, cy(astar) - L2 / 2, c3r + GAP_C + bw("out"), cy(astar) + L2 / 2)
    EL.append(dict(kind="box", name="out", rect=out, fill="oGreenS", text=T["out"]))
    if out[2] > W:
        raise SystemExit(f"flow figure too wide: {out[2]:.1f} mm > {W} mm")
    bottom = max(e["rect"][3] for e in EL) + PAD
    for name, l, r, title, ours in (("S1", c1l, c1r, "Per-image questions", False),
                                    ("S2", c2l, c2r, "Metric hypothesis memory", True),
                                    ("S3", c3l, c3r, "Search and control", False)):
        EL.append(dict(kind="container", name=name, rect=(l, TOP, r, bottom), ours=ours))
        zone(f"t_{name}", (l + 0.6, 0.4, r, 4.0), title, SIZES["title"], align="left", italic=True, bold=True)
    # branch labels
    yes_no(weak, "r", "yes"); yes_no(weak, "b", "no")
    yes_no(commit, "r", "yes"); yes_no(commit, "b", "no")
    # wires
    arrow([(w_in, cy(vlm)), (vlm[0], cy(vlm))])
    arrow([(x1, vlm[3]), (x1, weak[1])], "oBlueD")
    arrow([(weak[2], cy(weak)), (crop[0], cy(weak))], "oBlueD")
    arrow([(x1, weak[3]), (x1, weight[1])], "oBlueD")
    yin = weight[1] + 1.6
    arrow([(cx(crop), crop[3]), (cx(crop), yin), (weight[2], yin)], "oBlueD")
    j1, yout = (c1r + c2l) / 2, weight[3] - 1.6
    arrow([(weight[2], yout), (j1, yout), (j1, cy(lift)), (lift[0], cy(lift))])
    arrow([(xa, lift[3]), (xa, fuse[1])], "oOrangeD")
    arrow([(lift[2], cy(lift)), (xb, cy(lift)), (xb, new[1])], "oOrangeD")
    arrow([(xa, fuse[3]), (xa, neg[1])], "oOrangeD")
    arrow([(xb, new[3]), (xb, neg[1])], "oOrangeD")
    j2 = c2r + 0.8  # leave a long final run into the diamond
    arrow([(neg[2], cy(neg)), (j2, cy(neg)), (j2, cy(commit)), (commit[0], cy(commit))])
    arrow([(commit[2], cy(commit)), (goto[0], cy(commit))], "oGreyD")
    arrow([(x3, commit[3]), (x3, els[1])], "oGreyD")
    arrow([(xr, goto[3]), (xr, astar[1])], "oGreyD")
    arrow([(x3, els[3]), (x3, cy(astar)), (astar[0], cy(astar))], "oGreyD")
    arrow([(astar[2], cy(astar)), (out[0], cy(astar))])
    yl = bottom + 2.0
    zone("loop_lab", ((W - 18) / 2, yl - 1.25, (W + 18) / 2, yl + 1.25), "next frame", SIZES["caption"], fill="white")
    arrow([(cx(out), out[3]), (cx(out), yl), (w_in / 2, yl), (w_in / 2, EL[0]["rect"][3])], under=["loop_lab"])
    return yl + 1.3


def p(x, y):
    return f"({x:.2f},{-y:.2f})"


def tikz(H):
    out = ["\\begin{tikzpicture}[x=1mm, y=1mm, >={Stealth[length=1.4mm, width=1.3mm]}]",
           f"\\useasboundingbox (0,0) rectangle {p(W, H)};"]
    for e in sorted(EL, key=lambda e: e["kind"] != "container"):
        x0, y0, x1, y1 = e["rect"]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if e["kind"] == "container":
            st = "fill=oOrangeP, draw=oOrangeD, line width=0.8pt" if e["ours"] else "draw=oGreyK, line width=0.36pt"
            out.append(f"\\draw[{st}, dash pattern=on 3.2pt off 2.2pt, rounded corners=5pt] {p(x0, y0)} rectangle "
                       f"{p(x1, y1)};")
        elif e["kind"] == "box":
            out.append(f"\\filldraw[fill={e['fill']}, draw=black, line width=0.36pt, rounded corners=2.6pt] "
                       f"{p(x0, y0)} rectangle {p(x1, y1)};")
            out.append(f"\\node[font={font(SIZES['label'])}, align=center, inner sep=0, anchor=center] at {p(mx, my)} "
                       f"{{{e['text']}}};")
        elif e["kind"] == "dia":
            out.append(f"\\filldraw[fill={e['fill']}, draw=black, line width=0.36pt] {p(mx, y0)} -- {p(x1, my)} -- "
                       f"{p(mx, y1)} -- {p(x0, my)} -- cycle;")
            out.append(f"\\node[font={font(SIZES['label'])}, align=center, inner sep=0, anchor=center] at {p(mx, my)} "
                       f"{{{e['text']}}};")
        elif e["kind"] == "zone":
            if e.get("fill"):
                out.append(f"\\fill[{e['fill']}] {p(x0, y0)} rectangle {p(x1, y1)};")
            anchor, at = {"left": ("west", p(x0, my)), "center": ("center", p(mx, my))}[e["align"]]
            out.append(f"\\node[font={font(e['size'], e['bold'], e['italic'])}, text width={x1 - x0:.2f}mm, "
                       f"align={e['align']}, inner sep=0, anchor={anchor}] at {at} {{{e['text']}}};")
    wires = [f"\\draw[->, line width=0.6pt, draw={a['color']}] " + " -- ".join(p(x, y) for x, y in a["pts"]) + ";"
             for a in ARROWS]
    n_cont = 2 + sum(1 for e in EL if e["kind"] == "container")
    out[n_cont:n_cont] = wires  # above the containers, below nodes and label patches
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def in_node(e, x, y, m=0.0):
    x0, y0, x1, y1 = e["rect"]
    if e["kind"] == "dia":
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        return abs(x - (x0 + x1) / 2) / max(hw - m, 1e-6) + abs(y - (y0 + y1) / 2) / max(hh - m, 1e-6) <= 1.0
    return x0 + m <= x <= x1 - m and y0 + m <= y <= y1 - m


def audit(body, H):
    bad = []
    used = set(re.findall(r"o(?:Blue|Orange|Green|Grey)[A-Z]", body)) | {a["color"] for a in ARROWS}
    if [c for c in used if c not in OFFICE and c != "black"]:
        bad.append(f"off-palette colours: {used}")
    widths = set(re.findall(r"line width=([\d.]+)pt", body))
    if len(widths) > 4:
        bad.append(f"stroke-zoo: {sorted(widths)}")
    if min(float(s) for s in re.findall(r"\\fontsize\{([\d.]+)\}", body)) < 6.5:
        bad.append("text below 6.5 pt")
    # every symbol (a letter with its sub/superscript) used in a node must also appear in the caption
    atom = r"\\[a-zA-Z]+_\{[^}]*\}|\\[a-zA-Z]+_[a-zA-Z]|[a-zA-Z]_\{[^}]*\}|[a-zA-Z]_[a-zA-Z]|\\(?:lambda|gamma|pi|mu|tau)"
    atoms = lambda txt: {a for m in re.findall(r"\$([^$]+)\$", txt) for a in re.findall(atom, m)}
    missing = sorted(atoms("".join(T.values())) - atoms(CAPTION))
    single = lambda txt: set(re.findall(r"\$([a-zA-Z])\$", txt))
    missing += sorted(single("".join(T.values())) - single(CAPTION))
    if missing:
        bad.append(f"undefined symbols (not in the caption): {missing}")
    nodes = [e for e in EL if e["kind"] in ("box", "dia")]
    leaves = [e for e in EL if e["kind"] != "container"]
    # one path from the input to the action
    host = lambda pt: next((e["name"] for e in nodes if in_node(e, pt[0], pt[1], -0.05) or
                            (e["rect"][0] - 0.05 <= pt[0] <= e["rect"][2] + 0.05 and
                             e["rect"][1] - 0.05 <= pt[1] <= e["rect"][3] + 0.05)), None)
    graph = {}
    for a in ARROWS:
        u, v = host(a["pts"][0]), host(a["pts"][-1])
        if u and v:
            graph.setdefault(u, set()).add(v)
    seen, todo = {"in"}, ["in"]
    while todo:
        for n in graph.get(todo.pop(), ()):
            if n not in seen:
                seen.add(n); todo.append(n)
    for n in (e["name"] for e in nodes):
        if n not in seen:
            bad.append(f"unreachable from the input: {n}")
    for a in ARROWS:
        if len(a["pts"]) > 2 and np.hypot(*np.subtract(a["pts"][-1], a["pts"][-2])) < 10 * PT:
            bad.append(f"short final segment: arrow ending at {a['pts'][-1]}")
        for (xa, ya), (xb, yb) in zip(a["pts"], a["pts"][1:]):
            if abs(xa - xb) > 1e-6 and abs(ya - yb) > 1e-6:
                bad.append(f"diagonal edge at {(xa, ya)}")
            n = max(2, int(np.hypot(xb - xa, yb - ya) / 0.2))
            for t in np.linspace(0, 1, n)[1:-1]:
                x, y = xa + t * (xb - xa), ya + t * (yb - ya)
                hit = [e["name"] for e in leaves if e["name"] not in a["under"] and in_node(e, x, y, 0.3)]
                if hit:
                    bad.append(f"arrow {a['pts'][0]} -> {a['pts'][-1]} through {hit[0]}")
                    break
    segs = [(a_i, s) for a_i, a in enumerate(ARROWS) for s in zip(a["pts"], a["pts"][1:])]
    for i, (ai, ((x0, y0), (x1, y1))) in enumerate(segs):  # no two wires run on top of each other
        for aj, ((u0, v0), (u1, v1)) in segs[i + 1:]:
            if ai == aj:
                continue
            if abs(y0 - y1) < 1e-6 and abs(v0 - v1) < 1e-6 and abs(y0 - v0) < 0.4:
                if min(max(x0, x1), max(u0, u1)) - max(min(x0, x1), min(u0, u1)) > 0.3:
                    bad.append(f"wires overlap at y={y0:.1f}")
            if abs(x0 - x1) < 1e-6 and abs(u0 - u1) < 1e-6 and abs(x0 - u0) < 0.4:
                if min(max(y0, y1), max(v0, v1)) - max(min(y0, y1), min(v0, v1)) > 0.3:
                    bad.append(f"wires overlap at x={x0:.1f}")
    for i, a in enumerate(leaves):
        for b in leaves[i + 1:]:
            ra, rb = a["rect"], b["rect"]
            if min(ra[2], rb[2]) - max(ra[0], rb[0]) > 0.05 and min(ra[3], rb[3]) - max(ra[1], rb[1]) > 0.05:
                bad.append(f"overlap: {a['name']} / {b['name']}")
        r = a["rect"]
        if r[0] < -0.01 or r[1] < -0.01 or r[2] > W + 0.01 or r[3] > H + 0.01:
            bad.append(f"off canvas: {a['name']}")
    for c in (e for e in EL if e["kind"] == "container"):
        for e in nodes:
            q, r = e["rect"], c["rect"]
            if r[0] <= cx(q) <= r[2] and not (q[0] >= r[0] + 1.5 and q[2] <= r[2] - 1.5 and q[1] >= r[1] + 1.5
                                               and q[3] <= r[3] - 1.5):
                bad.append(f"{e['name']} too close to the edge of {c['name']}")
    pw, words = compile_tex(body, "flow_test")
    if abs(pw - W) > 0.3:
        bad.append(f"page width {pw:.1f} mm != {W}")
    for x0, y0, x1, y1, t in words:
        hosts = [e for e in leaves if in_node(e, (x0 + x1) / 2, (y0 + y1) / 2)]
        if not hosts:
            bad.append(f"text off node: {t!r}")
            continue
        e = hosts[0]
        if e["kind"] == "box":
            r = e["rect"]
            if not (x0 >= r[0] + 1.2 and x1 <= r[2] - 1.2 and y0 >= r[1] + 0.4 and y1 <= r[3] - 0.4):
                bad.append(f"text overflows {e['name']}: {t!r}")
        elif e["kind"] == "dia":
            if not all(in_node(e, x, y, 0.8) for x in (x0, x1) for y in (y0, y1)):
                bad.append(f"text overflows diamond {e['name']}: {t!r}")
        elif not all(e["rect"][0] - 0.05 <= x <= e["rect"][2] + 0.05 for x in (x0, x1)):
            bad.append(f"text overflows zone {e['name']}: {t!r}")
    multi = [w for w in words if len(w[4]) > 1]
    for i, a in enumerate(multi):
        for b in multi[i + 1:]:
            if min(a[2], b[2]) - max(a[0], b[0]) > 0.3 and min(a[3], b[3]) - max(a[1], b[1]) > 0.3:
                bad.append(f"words overlap: {a[4]!r} / {b[4]!r}")
    bad += raster_check(os.path.join("/tmp", "flow_test", "flow_test.pdf"), EL, ARROWS, "flow")
    bad = sorted(set(bad))
    if bad:
        raise SystemExit("flow figure audit failed:\n  " + "\n  ".join(bad))
    print(f"flow figure: audit ok ({len(words)} words, {len(nodes)} nodes, {len(ARROWS)} arrows, {W:.0f}x{H:.1f} mm)")


def main():
    tw = measure()
    H = layout(tw)
    body = tikz(H)
    audit(body, H)
    tex = (f"\\begin{{figure*}}[t]\n\\centering\n{COLORS}\n{body}\n\\caption{{{CAPTION}}}\n\\label{{fig:mech}}\n"
           "\\end{figure*}\n")
    open(os.path.join(LATEX, "sec", "fig_mech.tex"), "w").write(tex)
    print("wrote sec/fig_mech.tex")


if __name__ == "__main__":
    main()
