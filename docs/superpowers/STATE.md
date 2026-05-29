# PROJECT STATE — QOT course restructure

> **Single source of truth for "where are we?"** Update this file at the end of every work
> session. Read it (plus the design spec) at the start of every session, especially after a
> compaction / `/clear` / new conversation.

**Last updated:** 2026-05-28
**Phase:** Design complete → next is the build plan.
**Canonical structure:** `docs/superpowers/specs/2026-05-28-qot-course-restructure-design.md`
**Audit (rationale):** `docs/audit-notebooks-2026-05-28.md`
**Standing rules:** root `CLAUDE.md`

---

## Next action

Write the **build plan** (writing-plans skill) — per-notebook contracts for the ~57 notebooks
(concept, prerequisites, required contents, source cells to mine, synthesis link). Then finalize
the graphic charter (palette module), then reorganize folders, then build one notebook at a time.

## Locked decisions

- D1 Module folders `00_GettingStarted`/`01_…`/`02_…`/`03_…`/`04_…`, plain `.ipynb` (NO Jupyter Book).
- D2 Per-module numbering `NN_snake_title.ipynb`.
- D3 Grain: **one concept = one notebook**; bricks → synthesis (≈ a current sNN).
- D4 Decide counts before moving files.
- D5 Capstone = **A+B+C** multi-notebook arc (naive → diagnose redundancy → enrich embedding → discriminate → synthesize).
- D6 Total ≈ **57** notebooks.
- D7 Apply PPSP house style (palette centralized, no hardcoded hex, fixed fig dims), adapted to matplotlib.

## Open questions

- Charter: keep status emoji (🔬) in headers or align with PPSP "no emojis"? — DECIDE in charter phase.
- Exact QOT palette definition — DECIDE in charter phase.

## Progress checklist

### Done
- [x] Full audit of the 16 notebooks (`docs/audit-notebooks-2026-05-28.md`).
- [x] Committed s06–s16 (notebooks + modules + tests) — 12 per-session commits.
- [x] Reviewed PPSP house style (ConnectivityMetricsTutorials) — see memory `ppsp-tutorial-house-style`.
- [x] Brainstormed + locked the target structure (this STATE + the design spec).

### Not started — build (one notebook at a time, commit each, tick here)
- [ ] Build plan written (writing-plans) and reviewed.
- [ ] Charter finalized (palette module + `use_course_style`).
- [ ] Folders created + 16 current notebooks reorganized as seed syntheses.
- [ ] `00_GettingStarted` (2)
- [ ] `01_QuantumFoundations` (11)
- [ ] `02_InformationAndGeometry` (12)
- [ ] `03_ClassicalOptimalTransport` (14)
- [ ] `04_QuantumOptimalTransport` (18) — capstone arc 11–16 LAST and most careful.

## Coherence anchors (do not drift)

- The classical↔quantum **dictionary** thread runs through every module; each synthesis adds its row.
- The **3 QOT objects** (Bures-bridge / coupling-SDP / entropic) must stay reconciled (notebook 04/09).
- Keep tests green (`uv run pytest` → 161 passing baseline); validate numbers against closed forms.
- The capstone must **demonstrate** when QOT beats PLV (A+B+C), never hand-wave "it's open".
