# QOT Course — Restructure Design Spec

**Date:** 2026-05-28
**Status:** ✅ Design approved (brainstorming) — awaiting build plan
**Origin:** the notebook audit `docs/audit-notebooks-2026-05-28.md`
**Companion docs:** `docs/superpowers/STATE.md` (live progress), root `CLAUDE.md` (standing rules)

---

## 0. How to resume this work (READ FIRST after any compaction / clear / new conversation)

This project is a **course restructure in progress**. The quality regression in the old
s15/s16 was caused by losing working context mid-build. To prevent recurrence:

1. **This spec** is the canonical *what* — the target structure, the principles, the mapping.
2. **`docs/superpowers/STATE.md`** is the live *where-are-we* — read it to know the next action.
3. **Root `CLAUDE.md`** carries the non-negotiable rules, auto-loaded every session.

**The protocol:** build **one notebook at a time**, **commit per notebook**, **update STATE.md**
after each. Never bulk-generate many notebooks in one unreviewed pass. Never conclude "it
doesn't work" without having run the discriminating experiment.

---

## 1. Purpose & goals

Transform the course from a **flat set of 16 dense session-notebooks** into a **module-foldered,
fine-grained** structure where **one notebook = one concept**, concept-notebooks ("bricks")
build progressively to a **synthesis notebook** (≈ one of the current 16), and a consistent
**graphic charter** is applied. Driven by two findings from the audit:

- Several notebooks go **too fast** for a ~1h live session with a mixed-background audience
  (excellent as dense reads, overloaded as taught sessions).
- The **capstone (s15)** concluded "does QOT beat PLV? — open" on a **non-test**: its
  phase-qubit embedding makes QMI/Bures monotone reparametrizations of PLV (Spearman ≈ 1.0).

**Not in scope:** converting to Jupyter Book / mystmd. We stay on plain `.ipynb` in folders.

## 2. Locked decisions (from the 2026-05-28 brainstorming)

| # | Decision |
|---|----------|
| D1 | Folder structure by module: `00_GettingStarted`, `01_…`, `02_…`, `03_…`, `04_…`. Plain `.ipynb`. |
| D2 | **Per-module numbering** `NN_snake_title.ipynb` (reset within each folder). |
| D3 | **Grain locked: one concept = one notebook.** Bricks build to a synthesis (≈ current notebook). |
| D4 | Decide target counts **before** moving files (so we land in the final structure once). |
| D5 | **Capstone = A + B + C** as a multi-notebook research arc (naive → diagnose → enrich → discriminate → synthesize). |
| D6 | Total ≈ **57 notebooks** (16 → ~57, ~3.5×). Accepted as a large but coherent build. |
| D7 | Apply the PPSP house style (graphic charter), adapted to matplotlib notebooks (no Jupyter Book). |

## 3. Target architecture (~57 notebooks)

Legend: ◀ = synthesis (≈ a current sNN) · ★ = new (audit-driven).

### `00_GettingStarted` (2)
```
01_environment_setup            — uv, qiskit, IBM account, "does it run?"
02_what_is_optimal_transport    — OT intuition + first ot.emd2 + the trip map        ◀ (≈ S1)
```

### `01_QuantumFoundations` (11)
```
arc Qubit:
  01_amplitudes_and_superposition — complex amplitudes, |0>/|1>/|+>, norm
  02_the_bloch_sphere             — pure qubit as a point (θ,φ)
  03_measurement_and_born_rule    — amplitude → probability, Z-measurement
  04_shot_noise_and_first_circuit — sampling, convergence, first Qiskit run           ◀ (≈ S2)
arc Density:
  05_from_states_to_density_matrix— vector → operator, ρ, diagonal vs coherence
  06_pure_vs_mixed                — |+> vs I/2 (THE seed), purity, von Neumann entropy
  07_fidelity_and_trace_distance  — comparing states; the Bloch ball interior
  08_tomography_of_a_noisy_qubit  — reconstruct ρ → real states are mixed             ◀ (≈ S3)
arc Composite:
  09_tensor_and_partial_trace     — combine systems; partial trace = marginal
  10_entanglement                 — Bell state, pure whole / mixed parts
  11_quantum_channels             — CPTP/Kraus, depolarizing toward I/2               ◀ (≈ S4)
```

### `02_InformationAndGeometry` (12)
```
arc Classical info:
  01_shannon_entropy · 02_kl_divergence · 03_mutual_information · 04_transfer_entropy ◀ (≈ S5)
arc Geometry:
  05_the_probability_simplex · 06_fisher_rao_metric · 07_fisher_rao_geodesics
  08_two_geometries_one_simplex                                                       ◀ (≈ S6)
arc Quantum info:
  09_umegaki_relative_entropy · 10_quantum_mutual_information
  11_negative_conditional_entropy · 12_bures_distance                                 ◀ (≈ S7)
```

### `03_ClassicalOptimalTransport` (14)
```
arc Monge→Kantorovich:
  01_the_monge_problem · 02_where_monge_breaks · 03_the_kantorovich_lp
  04_birkhoff_and_assignment                                                          ◀ (≈ S8)
arc Wasserstein:
  05_wasserstein_distance · 06_1d_quantile_closed_form · 07_mccann_geodesic           ◀ (≈ S9)
arc Duality/Sinkhorn:
  08_kantorovich_duality · 09_entropic_regularization · 10_sinkhorn_algorithm
  11_amari_bridge                                                                     ◀ (≈ S10)
arc Gaussians:
  12_bures_wasserstein · 13_the_bures_bridge · 14_benamou_brenier_otto                ◀ (≈ S11)
```

### `04_QuantumOptimalTransport` (18)
```
arc Why:
  01_diagonal_collapse · 02_noncommuting_same_diagonal · 03_trevisan_taxonomy         ◀ (≈ S12)
arc Coupling SDP:
  04_quantum_coupling_sdp · 05_solving_qot_in_cvxpy                                   ◀ (≈ S13)
arc Quantum Sinkhorn:
  06_entropic_qot_gibbs
  07_operator_sinkhorn_iteration  ★ the actual algorithm (never run in old S14)
  08_quantum_amari_bridge                                                             ◀ (≈ S14)
★ 09_the_qot_zoo                  — reconcile Bures-bridge(S11) vs coupling-SDP(S13) vs entropic(S14): when equal? (audit A3)
★ 10_embedding_signals_into_states— the menu: phase / amplitude / covariance / multi-freq, and what each preserves
arc Capstone (A+B+C):
  11_kuramoto_dyad_and_ground_truth   — synthetic dyad, known injected K, PLV/corr baselines
  12_naive_phase_embedding_tracks_K   — naive phase-qubit: all measures rise with K ("looks like it works")
  13_the_redundancy_reveal      [C]   — but Spearman(PLV,QMI)≈1: quantum measures = PLV in disguise → can't win, by construction
  14_richer_embedding_applied   [B]   — enrich the embedding (mixed/amplitude/multi-freq) → ρ_AB encodes more than |PLV|
  15_discriminating_experiment  [A]   — same phase-diff distribution, different higher-order structure → QOT separates where PLV is blind
  16_capstone_synthesis               — when QOT helps, at what statistical cost, honest caveats
arc Frontier:
  17_vqe_limitation_de_palma · 18_frontier_synthesis                                  ◀ (≈ S16)
```

**Counts:** 00:2 · 01:11 · 02:12 · 03:14 · 04:18 = **57**.

## 4. Old → new mapping (source material to mine)

| Current | Feeds new notebooks |
|---|---|
| s01 roadmap | 00/01–02 |
| s02 qubits | 01/01–04 |
| s03 density | 01/05–08 |
| s04 composite | 01/09–11 |
| s05 information | 02/01–04 |
| s06 geometry | 02/05–08 |
| s07 quantum_infotheory | 02/09–12 |
| s08 monge_kantorovich | 03/01–04 |
| s09 wasserstein | 03/05–07 |
| s10 sinkhorn | 03/08–11 |
| s11 gaussians | 03/12–14 |
| s12 why_qot | 04/01–03 |
| s13 sdp | 04/04–05 |
| s14 quantum_sinkhorn | 04/06–08 (+07 NEW: real operator iteration) |
| s15 capstone | 04/11–16 (REBUILT as A+B+C; the redundancy is now a *result*, not a dead end) |
| s16 frontier_synthesis | 04/17–18 |
| (none — NEW) | 04/09 qot_zoo, 04/10 embeddings |

The src modules (`quantum/`, `infotheory/`, `transport/`, `quantum_ot/`, `geometry/`) and the
161 passing tests are **kept and reused** — they are validated. Restructuring is at the
notebook layer; src changes are limited to the charter (palette) + the new capstone/embedding code.

## 5. The brick → synthesis principle

- A **brick** teaches exactly one concept: intuition → minimal implementation → "Read the figure".
  Short (~30–45 min), self-contained, one new idea.
- A **synthesis** assumes its bricks, integrates them, delivers the punchline + the dictionary
  row + (where relevant) the hardware/application demo. It is *lighter* than the old dense
  notebook because it no longer teaches every concept from scratch.
- Bricks within an arc declare their prerequisites (the dependency graph, à la
  `ConnectivityMetricsTutorials/docs/CONTEXT.md`).

## 6. Graphic charter (target — adapted from the PPSP house style)

Model: `ConnectivityMetricsTutorials/docs/STYLE_GUIDE.md` + `src/colors.py`, adapted to plain
matplotlib notebooks (no MyST/Jupyter Book). Target rules:

- **Centralized palette**: upgrade `qot_course/viz.py` (currently `SOURCE_COLOR`/`TARGET_COLOR`/
  `FLOW_COLOR` + `use_course_style()`) into a proper `COLORS`-style module. **No hardcoded hex in
  notebooks** (clean the existing `#0ea5e9`, `#475569`, `#94a3b8` in s09/s11/s15).
- **Fixed figure dimensions + DPI**, fixed font sizes (per STYLE_GUIDE table).
- **Every figure explained** ("Read the figure") — already a QOT strength, keep it.
- **Cite source papers** in docstrings and notebook references — already a QOT strength.
- **Open charter decisions (finalize in the "apply charter" phase):**
  (a) status emoji in headers (current QOT uses 🔬) — align with PPSP "no emojis" or keep?
  (b) the exact QOT palette (transport source/target/flow + density-matrix Re/Im + band-like roles).

## 7. Build order (phasing)

1. **Write the build plan** (writing-plans) — per-notebook contracts (concept, prereqs, contents,
   source cells, synthesis link).
2. **Finalize the charter** (palette module + `use_course_style`), since every notebook uses it.
3. **Create folders + reorganize** existing 16 into the target tree (as the seed syntheses).
4. **Build module by module**, one notebook at a time, commit per notebook, update STATE.md.
   Suggested order: 00 → 01 → 02 → 03 → 04, with the **capstone arc (04/11–16)** last and most careful.

## 8. References

- Audit: `docs/audit-notebooks-2026-05-28.md`
- Curriculum intent: `docs/qot-course-session-by-session.md`, `docs/superpowers/specs/2026-05-27-qot-course-design.md`
- House style source: `/Users/remyramadour/Workspace/PPSP/Sandbox/Workshops/ConnectivityMetricsTutorials` (`docs/CONTEXT.md`, `docs/STYLE_GUIDE.md`)
