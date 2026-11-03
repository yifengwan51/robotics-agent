#!/usr/bin/env bash
# Rebuild every number, table and figure of the paper from the saved TEST records, compile it, and check it.
#   bash scripts/build_paper.sh        -> paper/latex/paper.pdf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
~/miniforge3/envs/vlm/bin/python scripts/paper_assets.py
~/miniforge3/envs/embnav/bin/python scripts/make_paper_figures.py
~/miniforge3/envs/embnav/bin/python scripts/make_arch_figure.py
~/miniforge3/envs/embnav/bin/python scripts/make_trajectory_figure.py
cd paper/latex
~/miniforge3/envs/tex/bin/tectonic -X compile main.tex --keep-logs --outdir /tmp/s2m_build > /tmp/s2m_build.log 2>&1
grep -iE "undefined|Overfull" /tmp/s2m_build.log | grep -v lineno | sort -u || true
cp /tmp/s2m_build/main.pdf paper.pdf
echo "paper.pdf: $(pdfinfo paper.pdf | awk '/Pages/{print $2}') pages"
