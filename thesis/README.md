# MPhil thesis (ROAS, HKUST(GZ)): Ask Small VLMs Only What They Can See

Built on [hkustgz-thesis-template](https://github.com/crimson-and-clover/hkustgz-thesis-template) (MIT, `LICENSE`) and
checked against the FYTGS(GZ) *Guidelines on Thesis Preparation* (22 Jan 2024) and the MPhil sample pages (Mar 2024):
page order title, authorization, signature, acknowledgments, contents, lists, abstract, body, bibliography,
appendices; 12 pt, one-and-a-half spacing, 25 mm margins, centred page numbers (Roman before Chapter 1, title page
counted as i but not shown); English abstract of at most 300 words; IEEE references.

    bash ../scripts/build_paper.sh   # regenerate the paper's numbers, tables, figures
    bash sync_from_paper.sh          # copy them into chapters/generated, figures/, references.bib
    tectonic -X compile main.tex     # -> main.pdf

Chapters are written for the thesis in `chapters/`. Every number comes from generated files:
`chapters/generated/numbers.tex` (the paper's `scripts/paper_assets.py`), `thesis_numbers.tex` and the trace and
excerpt tables (`make_thesis_assets.py`), and `thesis_numbers_fig.tex` with the `figures/th_*.pdf` figures
(`make_thesis_figures.py`, env embnav). `fix_tables.py` adapts the paper's tables to the thesis page.
`references_thesis.bib` holds references used only by the thesis.

Template changes: minted not loaded (needs -shell-escape); biblatex BibTeX backend (no biber here); titles set as
ordinary text so they break across lines; title page without a visible number; signature page with names under the
signature lines and the Thrust name, as in the sample page.

Red text in the PDF marks what the author must fill in before submission (Thrust Head, dates, acknowledgments).
