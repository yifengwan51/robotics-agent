"""Adapt the paper's generated tables to the thesis page (run by sync_from_paper.sh after copying them).

- tab_main: the 95% CI column is empty for every reported result; the thesis gives the intervals of its own runs in
  a separate table, so the column is dropped here. The oracle row gets 'true position' instead of an empty model.
- tab_robothor: 'this work' instead of an empty year. tab_abl: the reference row says so.
- tables set for the paper's narrow column are stretched to the thesis text width.
"""
import os
import re
import sys

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chapters", "generated")


def edit(name, fn):
    p = os.path.join(G, name)
    s = open(p).read()
    open(p, "w").write(fn(s))


def main_table(s):
    s = s.replace("\\begin{tabular}{@{}llrrrrrr@{\\,}lc@{}}", "\\begin{tabular}{@{}llrrrrrr@{\\,}l@{}}")
    s = s.replace(" & 95\\% CI\\\\", "\\\\")
    s = s.replace("\\multicolumn{10}", "\\multicolumn{9}")
    s = re.sub(r" & (--|\[[-\d.]+, [-\d.]+\])\\\\\n", r"\\\\\n", s)
    s = s.replace("Oracle target$^\\ddagger$ & --", "Oracle target$^\\ddagger$ & true position")
    return s


def robothor(s):
    s = s.replace("\\ours (Qwen3-VL-4B) & RGB-D & -- &", "\\ours (Qwen3-VL-4B) & RGB-D & this work &")
    s = s.replace("\\centering\\footnotesize", "\\centering\\small").replace("\\setlength{\\tabcolsep}{6pt}\\renewcommand{\\arraystretch}{0.95}\n", "")
    return stretch(s)


def ablation(s):
    return s.replace("\\multicolumn{2}{c}{--} & --\\\\", "\\multicolumn{3}{c}{reference}\\\\")


def stretch(s):
    """tabular with only l/c/r columns -> tabular* of the full text width."""
    return re.sub(r"\\begin\{tabular\}\{@\{\}([lcr]+)@\{\}\}(.*?)\\end\{tabular\}",
                  lambda m: "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}" + m.group(1) + "@{}}" + m.group(2)
                  + "\\end{tabular*}", s, flags=re.S)


def traj_caption(s):
    """The thesis variant shows short panel titles; the caption says what each panel shows."""
    if "(a)~the target starts behind" in s:
        return s
    s = s.replace("\\label{fig:traj}", "", 1).replace("}\n\n", "}\n", 1)
    return s.replace("is a look-alike.}", "is a look-alike. Each map has its own scale. (a)~the target starts behind "
                     "the agent; (b)~furniture blocks the EB prompt; (c)~\\ours proposes only look-alikes.}\n\\label{fig:traj}", 1)


edit("fig_traj.tex", traj_caption)
edit("tab_main.tex", main_table)
edit("tab_robothor.tex", robothor)
edit("tab_abl.tex", ablation)
for n in ("supp_tab_probe.tex", "supp_tab_taxonomy.tex", "supp_tab_robothor.tex"):
    edit(n, stretch)
print("tables adapted")
