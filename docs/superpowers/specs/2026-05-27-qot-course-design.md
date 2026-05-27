# Quantum Optimal Transport — A Hands-On Course

**Design specification**

- **Date:** 2026-05-27
- **Status:** Design approved (pending spec review)
- **Owner:** Rémy Ramadour — PPSP lab, CHU Sainte-Justine
- **Repository branch:** `course` (orphan; the previous demo is preserved on `master` + `origin`)
- **Language of all deliverables:** English (code, notebooks, PDFs, glossary, dictionary)

---

## 1. Purpose & vision

A progressive, **hands-on course** that takes a scientifically literate but quantum-naive
audience from **classical optimal transport (OT)** to **quantum optimal transport (QOT)**,
built **concept by concept** over **16 weekly 2-hour workshops** (~3 months).

It must be **two things at once**:

1. **A correct, runnable demonstration of genuine QOT** — the operator-level,
   convex-optimization (SDP) formulation that the literature actually defines, validated
   numerically against reference implementations.
2. **Reusable pedagogical material** — functions, exhaustive notebooks, and typeset LaTeX
   summaries — that the lab can appropriate, extend, and teach from.

**Explicitly NOT:**

- A "quantum advantage on brains" claim, or any quantum hype. The brain (and most lab data)
  is a classical system; the quantum formalism is used where it is the *correct geometry for
  PSD / phase-space data*, not as a metaphor for speedup.
- A salvage of the previous `quantum-optimal-transport-demo`. That code conflated two
  unrelated notions of "QOT", implemented neither correctly, and is abandoned (table rase).
- A research paper — although the capstone deliberately seeds one.

## 2. Audience & prerequisites

- **Audience:** PPSP lab researchers and collaborators. The lab does **not** work only on the
  brain; the course is domain-agnostic, with a neuroscience/hyperscanning capstone.
- **Assumed:** comfort with Python + `numpy`/`scipy`; general scientific maturity.
- **Not assumed:** quantum mechanics, quantum computing, measure theory, or information theory
  beyond the basics. The quantum and information-theoretic material is built from zero.

## 3. Guiding principles

1. **Honesty over hype.** Three honesty anchors are taught explicitly: (a) the *diagonal-collapse
   lesson* — a classical distribution encoded as a diagonal density matrix makes QOT reduce to
   classical OT; (b) *quantum-inspired ≠ hardware advantage* — the QOT computation is classical;
   (c) *quantum transfer entropy is a research frontier*, not a settled definition.
2. **Information theory & geometry is the spine.** Relative entropy (KL / von Neumann) is the
   common parent of entropic-regularized OT (Sinkhorn), quantum Sinkhorn, and the capstone
   coupling measure. Information geometry (Fisher–Rao) and transport geometry (Wasserstein) are
   the two great geometries on the probability simplex; they meet at entropic regularization
   (Amari).
3. **Every concept follows the same arc:** intuition → mathematics → code → visualization →
   one-page typeset summary.
4. **Two living artifacts**, grown every session: a **classical↔quantum dictionary** and a
   **glossary**.
5. **Reproducibility & quality:** seeded randomness, tests against reference implementations,
   parameters in config, structured logging, no silenced output.

## 4. Scientific through-line

- **Core thread — OT on PSD-matrix data.** Whenever the data object is a positive-semidefinite
  matrix (covariance, structure tensor, diffusion tensor, coherency/polarization matrix, kernel),
  it *is* a (real or complex) density operator, and quantum information geometry — including QOT —
  is the rigorous geometry, not an analogy. The headline application is **Peyré, Chizat, Vialard
  & Solomon, "Quantum Optimal Transport for Tensor Field Processing"** (DTI, texture, meshing),
  which uses von Neumann entropy and a PSD-generalized Sinkhorn — exactly the machinery the
  course builds.
- **The "OT fails / QOT succeeds" exemplars.** (a) Clean & non-negotiable: a single qubit,
  `|+⟩` vs the maximally mixed state — identical computational-basis statistics (same diagonal),
  distinct at the operator level (coherence). (b) "Adult" version: two PSD tensors with identical
  eigenvalues but rotated eigenvectors — identical to spectrum-based scalar OT, distinct to QOT
  (non-commutativity = orientation).
- **Information-geometry bridge.** Sinkhorn (classical, S10) and quantum Sinkhorn (S14) are where
  Wasserstein geometry meets KL / Fisher–Rao geometry (Amari 2018; Wuchen Li).
- **Capstone (open research, S15).** Inter-system coupling for **dyads** (hyperscanning) framed
  as **quantum relative entropy to the decoupled (block-diagonal) state**, `D(ρ‖σ) = S(σ) − S(ρ)`,
  with the QOT coupling providing the bipartite joint state and barycenters. Validated on
  **synthetic Kuramoto dyads** where the injected coupling is known. Honest caveats baked in:
  it is a *measure on a classically-built density matrix*, must beat PLV/Euclidean baselines
  empirically, and requires shrinkage estimation + volume-conduction (imaginary-part) correction.
  - *The structural reason the dyadic setting earns its place:* a dyad is genuinely **bipartite**,
    and a quantum coupling (the QOT transport plan) is a bipartite density matrix with two
    marginals — the dyad's shape matches QOT's native object. **Caveat to teach explicitly:**
    stacking two brains' channels yields a *direct-sum* density matrix, **not** a tensor-product
    bipartite state; "quantum mutual information of the connectivity matrix" would be wrong. The
    correct data-driven measure is the relative entropy to the block-diagonal (decoupled) state.

## 5. Curriculum — 16 sessions

Movements: **M0** kickoff · **M1** quantum foundations · **M2** information theory & geometry
(the spine) · **M3** classical OT · **M4** quantum OT.

The **HW** column flags sessions with an optional real-quantum-hardware cell (graceful fallback
to a noisy simulator — see §9).

| # | Movement | Theme | Key concepts | HW |
|---|----------|-------|--------------|----|
| **S1** | M0 | Teaser & roadmap | The destination (a visual transport demo), the dictionary skeleton, course mechanics, env setup, first `ot.emd2` | — |
| **S2** | M1 | Qubits & states | Complex amplitudes, Bloch sphere, measurement, Born rule (Clifford/Pauli nod) | ✓ |
| **S3** | M1 | Density matrices | Pure vs mixed, PSD & trace-1, von Neumann entropy, purity, fidelity, trace distance | ✓ |
| **S4** | M1 | Composite systems & channels | Tensor product, partial trace, entanglement, quantum channels (Kraus, CPTP), Stinespring | ✓ |
| **S5** | M2 | Classical information theory | Shannon entropy, **KL divergence**, mutual information, conditional MI, **transfer entropy** (Schreiber); PID (circumspect) | — |
| **S6** | M2 | Information geometry | Fisher–Rao metric, statistical manifold, exponential families, dual connections (Amari); the two geometries of the simplex | — |
| **S7** | M2 | Quantum information theory | von Neumann entropy, **Umegaki relative entropy (quantum KL)**, **quantum MI**, quantum conditional entropy (can be < 0), strong subadditivity; quantum Fisher (Bures/BKM); **quantum TE = frontier** | ○ |
| **S8** | M3 | Monge → Kantorovich | Maps vs couplings, marginals, the LP, Birkhoff polytope, assignment problem | — |
| **S9** | M3 | Wasserstein distances | `W_p`, the **1D closed form** (quantiles/sorting), metric properties, McCann/displacement interpolation | — |
| **S10** | M3 | Duality & Sinkhorn | Kantorovich duality, 1-Lipschitz dual (`W_1`), entropic regularization, Sinkhorn; **Amari: Sinkhorn = OT ∩ information geometry** | — |
| **S11** | M3 | Gaussians & dynamics | **Bures–Wasserstein** closed form on Gaussians (the hinge to quantum), Benamou–Brenier, Otto calculus | — |
| **S12** | M4 | Why QOT | Non-commutativity, what breaks, the **diagonal-collapse lesson** (plus-state vs maximally mixed); Trevisan's taxonomy; dictionary completed | — |
| **S13** | M4 | Coupling QOT = an SDP | Quantum couplings = bipartite `ρ` with partial-trace marginals, cost `tr(Cρ)`, SWAP cost → quantum `W_2`, solved with `cvxpy`; validated on qubits/Gaussians (Cole et al.) | ○ |
| **S14** | M4 | Quantum Sinkhorn | von Neumann–entropy regularization, dual, matrix-exp + partial-trace fixed point (Peyré tensor fields; Pelikh–Gerolin); **the quantum Amari bridge**; DTI/texture application | — |
| **S15** | M4 | Capstone (open problem) | Inter-system coupling = relative entropy to decoupled + QOT coupling, on a **synthetic Kuramoto dyad** (recover the injected coupling); honest caveats | — |
| **S16** | M4 | Frontier & synthesis | **VQA-limitations** cautionary tale (De Palma et al. 2023, closing the loop with the old demo); taxonomy recap (channels, qubit `W_1`, Carlen–Maas dynamic); open problems | ○ |

`✓` = hardware cell central to the session · `○` = optional hardware bridge · compressible to ~14
by merging S5+S6 and S15+S16 if group energy flags, but **16 is the target**.

## 6. Per-session deliverable contract ("definition of done")

Every session ships:

1. **A focused, reusable Python module** (typed, NumPy-style docstrings citing source papers),
   placed by *concept* (see §7), not by session number.
2. **An exhaustive teaching notebook** `notebooks/sNN_<topic>.ipynb`: intuition → derivation →
   code → **exercises to complete by participants** → visualization → sanity checks. Outputs
   cleared before commit.
3. **A typeset LaTeX PDF summary** `summaries/sNN_<topic>.pdf`, built from a reusable `.tex`
   template via `latexmk` (see §8).
4. **Appended entries** to `docs/dictionary.tex` (classical↔quantum correspondences) and
   `docs/glossary.tex` (term definitions).
5. **Tests** `tests/test_<module>.py` (see §11).

## 7. Repository architecture (proposed)

```
quantum-optimal-transport-demo/        # repo root (branch: course)
├── pyproject.toml                      # uv-managed; name -> qot-course
├── README.md                           # course overview + how to run
├── config/                             # YAML parameters (no hardcoded constants)
├── src/qot_course/
│   ├── quantum/        states.py · density.py · channels.py · tomography.py
│   ├── infotheory/     classical.py · geometry.py · quantum.py
│   ├── ot/             kantorovich.py · wasserstein.py · sinkhorn.py · gaussian.py
│   ├── qot/            coupling_sdp.py · quantum_sinkhorn.py · metrics.py · taxonomy.py
│   ├── capstone/       kuramoto.py · coupling.py
│   ├── hardware/       runtime.py            # QPU/sim backend with graceful fallback
│   └── viz/            plots.py · animations.py
├── notebooks/          s01_roadmap.ipynb … s16_frontier.ipynb
├── summaries/          template.tex · s01_*.tex/.pdf … (latexmk)
├── docs/
│   ├── dictionary.tex  glossary.tex         # living, cumulative
│   ├── references.bib
│   └── superpowers/specs/                   # this document
└── tests/              test_*.py
```

Modules are **concept-named and reusable** (modular-code principle); sessions compose them.
Notebooks and summaries are **session-indexed** to follow the course flow.

## 8. Tooling stack

- **Environment:** `uv` (`pyproject.toml` + `uv.lock`), Python 3.12+.
- **Core:** `numpy`, `scipy`, `matplotlib`.
- **Classical OT:** `POT` (`ot`).
- **Convex optimization (QOT SDP):** `cvxpy` (+ a solver, e.g. SCS/Clarabel).
- **Quantum:** `qiskit`, `qiskit-aer` (incl. noise models / fake backends),
  `qiskit-ibm-runtime` (real-hardware primitives).
- **PDF summaries:** native LaTeX via **MacTeX/`latexmk`** (verified present: TeX Live 2026,
  `pdflatex`/`xelatex`/`latexmk`/`dvisvgm`/`gs`), templated with `Jinja2`; figures from
  `matplotlib`.
- **Notebooks:** `jupyter`.
- **Quality:** `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`.

## 9. Hardware track

**Golden rule:** the QOT computation is **classical** (SDP, Sinkhorn, `numpy`/`cvxpy`); it never
runs on a QPU. Real hardware appears only where it adds honest pedagogical value:

- **M1 (its natural home).** S2: measure a real qubit → shot noise + readout error. S3:
  single-qubit **tomography on hardware → the reconstructed `ρ` is slightly mixed** (purity < 1) —
  the most honest motivation for density matrices. S4: a Bell state on hardware → entanglement
  degraded by noise = a quantum channel in action.
- **M4 (optional bridges).** S13: QOT between two **experimentally reconstructed** noisy states.
  S16: the naive VQE **on hardware** underperforms → reinforces the VQA-limitations lesson live.

**Engineering (robustness first).** Default everything to `AerSimulator` + **fake noisy backends**
(emulate a real device): offline, no token, no queue, reproducible. The few QPU cells use
`qiskit-ibm-runtime` (Sampler/Estimator) and **degrade gracefully** — automatic fallback to the
noisy simulator if no token / device unavailable. The instructor may pre-run one shared job.

**Feasibility (verified, 2026).** IBM **Open Plan**: 10 min QPU runtime / 28 days, 100+ qubit QPUs,
free with an account; **2026 promotion** (from March 16): use 20 min → unlock 180 min for 12 months.
Sufficient for a handful of tiny demonstrative jobs. *Caveats:* queue latency, noisy results, and
no live submission by 16 participants at once (batch or pre-run).

## 10. Living artifacts: dictionary & glossary

- `docs/dictionary.tex` — the **classical↔quantum correspondence table** (probability↔density
  matrix, marginal↔partial trace, Markov kernel↔channel, Shannon↔von Neumann, KL↔Umegaki
  relative entropy, MI↔quantum MI, coupling↔bipartite state, Sinkhorn↔quantum Sinkhorn, …),
  introduced empty in S1 and filled every session.
- `docs/glossary.tex` — **term definitions**, accumulated session by session.
- Both are English, typeset, and shipped as living PDFs alongside the summaries.

## 11. Quality, reproducibility & scientific rigor

- **Numerical correctness is tested against references**, not just "it runs": classical OT vs
  `POT`; the QOT SDP primal vs its dual and vs known closed forms (single qubit, Gaussians);
  quantum Sinkhorn vs the exact SDP as the regularization `ε → 0`. (This is precisely the test
  the old demo lacked.)
- **Seeded randomness**; parameters in `config/*.yaml`; structured `logging` (no silenced output,
  no hidden progress bars).
- **Docstrings cite source papers**; heuristic vs theoretically grounded choices are flagged.
- **Honesty disclaimers** are first-class content, not footnotes (see §3).

## 12. Repository / git strategy

- Work on the **orphan branch `course`** of the existing repo. The previous demo is preserved on
  `master` (local) and `origin` (GitHub `Ramdam17/quantum-optimal-transport-demo`); table rase is
  therefore non-destructive and reversible.
- The repo may be renamed (e.g. to `quantum-optimal-transport-course`) when published; `origin`
  is not force-overwritten until explicitly decided.

## 13. Risks & open questions

- **Capstone B is genuine research**, not a settled result: it must be validated empirically
  against PLV / Euclidean baselines, and depends on robust CSD estimation (shrinkage) and
  volume-conduction correction. Framed as an *open problem the participants now have the tools
  to attack*.
- **Hardware queue/latency** for a live workshop — mitigated by simulator-first design and
  pre-run jobs.
- **Quantum transfer entropy** has no canonical definition — taught as a frontier, with the same
  circumspection as PID.
- **Per-session scope/time** (2h) — each session's "exercises to complete" must be calibrated so
  the core fits; overflow becomes optional/bonus.
- **LaTeX summary throughput** — 16 PDFs; mitigated by a single reusable template + automation.

## 14. Out of scope (YAGNI)

- Any claim of quantum hardware speedup for OT/QOT.
- A real neuroimaging data pipeline (the capstone uses synthetic Kuramoto dyads).
- Implementation of the Carlen–Maas dynamic quantum Benamou–Brenier (mentioned in S16 only).
- Dynamic (Benamou–Brenier) classical OT is taught conceptually (S11), not implemented in depth.
- The original demo's modules (all discarded).

## 15. References

- M. Wirth, *Four Lectures (and Some Bonus Material) on Quantum Optimal Transport* (2025).
- D. Trevisan, *Quantum optimal transport: an invitation*, Boll. Unione Mat. Ital. (2024).
- S. Cole, M. Eckstein, S. Friedland, K. Życzkowski, *On Quantum Optimal Transport*, arXiv:2105.06922 (2022).
- P. Pelikh, A. Gerolin, *Quantum Optimal Transport: Regularization and Algorithms*, OPT2025 (NeurIPS workshop).
- G. De Palma, D. Trevisan, *Quantum Optimal Transport with Quantum Channels*, Ann. Henri Poincaré (2021).
- G. De Palma, M. Marvian, D. Trevisan, S. Lloyd, *The quantum Wasserstein distance of order 1*, IEEE Trans. Inf. Theory (2021).
- G. Peyré, L. Chizat, F.-X. Vialard, J. Solomon, *Quantum Optimal Transport for Tensor Field Processing*, arXiv:1612.08731.
- S. Amari et al., *Information Geometry Connecting Wasserstein Distance and KL Divergence via the Entropy-Relaxed Transportation Problem*, Information Geometry (2018), arXiv:1709.10219.
- G. De Palma, M. Marvian, C. Rouzé, D. S. França, *Limitations of variational quantum algorithms: a quantum optimal transport approach*, PRX Quantum (2023).
- M. De Domenico, J. Biamonte, *Spectral Entropies as Information-Theoretic Tools for Complex Network Comparison*, PRX (2016); + *Scale-resolved spectral entropy of brain functional connectivity*, NeuroImage (2020).
- M. Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS (2013).
- G. Peyré, M. Cuturi, *Computational Optimal Transport*, Found. Trends ML (2019).
- T. Schreiber, *Measuring Information Transfer*, Phys. Rev. Lett. (2000).

---

*Next step: spec review by the owner, then the `writing-plans` skill to produce the
implementation plan (per-sprint, starting with S1 + project scaffolding).*
