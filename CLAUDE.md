# CLAUDE.md — Quantum Optimal Transport course

> Project-level rules. Global preferences in `~/.claude/CLAUDE.md` also apply (uv, ruff, never
> silence console output, ask before unsolicited changes, cite literature).

## ⚠️ READ FIRST — every session, especially after a compaction / `/clear` / new conversation

This is a **course restructure in progress**. Before doing anything, read, in order:

1. `docs/superpowers/STATE.md` — where we are + the next action.
2. `docs/superpowers/specs/2026-05-28-qot-course-restructure-design.md` — the canonical target structure.
3. `docs/audit-notebooks-2026-05-28.md` — why we are doing this.

A prior context loss (compaction/clear) is what produced the rushed, un-reviewed old s15/s16.
These docs exist so that never recurs. **If STATE.md and the spec disagree with your memory of
the plan, the docs win.**

## Non-negotiable rules

- **One concept = one notebook.** Short focused "bricks" build to a "synthesis" notebook
  (≈ one of the original 16). Never cram multiple new concepts into one notebook.
- **Build one notebook at a time. Commit per notebook. Update `STATE.md` after each.**
  NEVER bulk-generate many notebooks in one unreviewed pass (that is the original failure).
- **Graphic charter:** centralized palette in `qot_course.viz`; **no hardcoded hex in notebooks**;
  fixed figure dimensions/DPI/fonts; every figure followed by a "Read the figure" explanation.
- **The capstone (04/11–16) must follow the A+B+C arc and *demonstrate* when QOT beats PLV.**
  NEVER conclude "it doesn't work / it's open" without having run the discriminating experiment.
  (The old capstone's "open question" was a non-test: the embedding made QMI/Bures monotone in PLV.)
- **Coherence anchors:** the classical↔quantum dictionary thread; the 3 QOT objects stay
  reconciled (04/09); keep `uv run pytest` green (161-passing baseline); validate numbers
  against closed forms.
- **Notebooks are output-free in git** (clear outputs before committing) and didactic
  (intuition → implementation → interpretation), per `notebook-quality` standards.

## Numbering & folders

`notebooks/NN_ModuleName/MM_snake_title.ipynb` — module folders `00_GettingStarted`,
`01_QuantumFoundations`, `02_InformationAndGeometry`, `03_ClassicalOptimalTransport`,
`04_QuantumOptimalTransport`. Plain `.ipynb` — **not** Jupyter Book.
