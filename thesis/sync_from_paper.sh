#!/usr/bin/env bash
# Copy the generated assets of the paper (numbers, tables, figures, bibliography) into the thesis, so every number
# in the thesis comes from the same scripts/paper_assets.py run as the paper. Run after scripts/build_paper.sh.
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
P="$T/../paper/latex"
TEX="$HOME/miniforge3/envs/tex/bin/tectonic"
cp "$P"/sec/numbers.tex "$P"/sec/tab_main.tex "$P"/sec/tab_abl.tex "$P"/sec/tab_robothor.tex \
   "$P"/sec/supp_tab_probe.tex "$P"/sec/supp_tab_taxonomy.tex "$P"/sec/supp_tab_robothor.tex "$T/chapters/generated/"
cp "$P/sec/fig_traj.tex" "$T/chapters/generated/"
[ -f "$P/sec/tab_sens.tex" ] && cp "$P/sec/tab_sens.tex" "$T/chapters/generated/"  # written once the sensitivity runs finish
cp "$P"/fig/*.pdf "$P"/fig/*.png "$T/figures/"
cp "$P/main.bib" "$T/references.bib"
# full-width paper figures have thesis variants (thesis text width, larger text); the column figures are reused
( cd "$T/.." && THESIS=1 "$HOME/miniforge3/envs/embnav/bin/python" scripts/make_paper_figures.py >/dev/null \
  && THESIS=1 "$HOME/miniforge3/envs/embnav/bin/python" scripts/make_trajectory_figure.py >/dev/null )
cp "$P/fig/analysis_thesis.pdf" "$P/fig/trajectories_thesis.pdf" "$T/figures/"
# the two TikZ figures of the paper, compiled on their own so the thesis can scale them to its text width
W="$(mktemp -d)"
for f in fig_arch fig_mech; do
  body="$(python3 - "$P/sec/$f.tex" <<'PY'
import sys, re
s = open(sys.argv[1]).read()
i, j = s.index("\\definecolor"), s.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
print(s[i:j])
PY
)"
  cat > "$W/$f.tex" <<TEXEOF
\documentclass[border=0pt]{standalone}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage{amsmath,amssymb,graphicx,tikz,xspace}
\usetikzlibrary{positioning,arrows.meta,fit,backgrounds}
\graphicspath{{$P/}}
\newcommand{\ours}{S2M\xspace}
\input{$P/sec/numbers}
\begin{document}
$body
\end{document}
TEXEOF
  "$TEX" -X compile "$W/$f.tex" --outdir "$W" >/dev/null 2>&1 || { echo "failed: $f"; exit 1; }
  cp "$W/$f.pdf" "$T/figures/$f.pdf"
done
rm -rf "$W"
echo "synced numbers, tables, figures, and bibliography from the paper"
# generated figure files point at fig/; the thesis keeps figures in figures/ and uses single-column floats
sed -i 's#{fig/#{figures/#g; s#figure\*#figure#g; s#table\*#table#g; s#figures/trajectories.pdf#figures/trajectories_thesis.pdf#' "$T"/chapters/generated/*.tex
# tables of the paper were set for a 17.4 cm text block and are scaled to the thesis text width
python3 - "$T" <<'PY'
import re, sys
t = sys.argv[1]
for name in ("tab_main", "tab_abl"):
    p = f"{t}/chapters/generated/{name}.tex"
    s = open(p).read()
    if "\\resizebox" not in s:
        s = s.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}", 1)
        s = s.replace("\\end{tabular}", "\\end{tabular}}", 1)
    open(p, "w").write(s)
PY
python3 "$T/make_thesis_assets.py"
python3 "$T/fix_tables.py"
"$HOME/miniforge3/envs/embnav/bin/python" "$T/make_thesis_figures.py"
