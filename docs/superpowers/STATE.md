# PROJECT STATE — QOT course restructure

> **Single source of truth for "where are we?"** Update this file at the end of every work
> session. Read it (plus the design spec) at the start of every session, especially after a
> compaction / `/clear` / new conversation.

**Last updated:** 2026-05-28 (Plan A — charter foundation — complete)
**Phase:** Charter foundation done → next is Plan B (folder reorg).
**Canonical structure:** `docs/superpowers/specs/2026-05-28-qot-course-restructure-design.md`
**Audit (rationale):** `docs/audit-notebooks-2026-05-28.md`
**Standing rules:** root `CLAUDE.md`

---

## Next action

**Execute Plan B — folder reorg**: create `00_GettingStarted`..`04_QuantumOptimalTransport`
module folders, `git mv` the 16 current notebooks in as seed syntheses, fix cross-refs.
Palette visual sign-off is pending — run `uv run python scripts/show_palette.py` first.

## Build roadmap (phased plans — each its own `docs/superpowers/plans/2026-05-28-*.md`)

- **Plan A — charter foundation** ✍️ written (this is the next to execute).
- **Plan B — folder reorg** (create `00..04`, `git mv` the 16 in as seed syntheses, fix cross-refs).
- **Plans C–G — per module** (00, 01, 02, 03, 04): bricks→syntheses, one notebook at a time.
- **Plan H — open-source meta** (README, CONTRIBUTING, LICENSE = **MIT code + CC-BY-4.0 content**; warm README/CONTRIBUTING in the charter voice).

## Locked decisions

- D1 Module folders `00_GettingStarted`/`01_…`/`02_…`/`03_…`/`04_…`, plain `.ipynb` (NO Jupyter Book).
- D2 Per-module numbering `NN_snake_title.ipynb`.
- D3 Grain: **one concept = one notebook**; bricks → synthesis (≈ a current sNN).
- D4 Decide counts before moving files.
- D5 Capstone = **A+B+C** multi-notebook arc (naive → diagnose redundancy → enrich embedding → discriminate → synthesize).
- D6 Total ≈ **57** notebooks.
- D7 Apply PPSP house style (palette centralized, no hardcoded hex, fixed fig dims), adapted to matplotlib.
- D8 **Voice = warm, empowering, celebratory** (audience incl. a self-love-affirmation learner): celebrate accomplishment, frame difficulty as growth, ban "obviously/simply/trivially/just"; **no decorative emojis**; rigor intact.
- D9 License: **MIT (code) + CC-BY-4.0 (content)** — open with attribution (applied in Plan H).

## Open questions

- _(none — palette signed off by Rémy 2026-05-28; charter fully locked.)_

## Progress checklist

### Done
- [x] Full audit of the 16 notebooks (`docs/audit-notebooks-2026-05-28.md`).
- [x] Committed s06–s16 (notebooks + modules + tests) — 12 per-session commits.
- [x] Reviewed PPSP house style (ConnectivityMetricsTutorials) — see memory `ppsp-tutorial-house-style`.
- [x] Brainstormed + locked the target structure (this STATE + the design spec).
- [x] Spec reviewed & approved by Rémy; charter voice/tone + no-emoji locked (D8).
- [x] Build plan decomposed (A–H); **Plan A (charter) written** and committed.

### Not started — build (one notebook at a time, commit each, tick here)
- [ ] Plans B–H written (Plan A done — see Build roadmap above).
- [x] Charter finalized (palette module + `use_course_style`)
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
