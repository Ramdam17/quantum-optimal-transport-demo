# S3 — Density Matrices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Session 3 — the density matrix ρ as the central object: pure vs mixed, purity, von Neumann entropy, fidelity & trace distance, the Bloch *ball*, and the pivotal lesson that `|+⟩` and the maximally mixed state share a diagonal but differ in their off-diagonal *coherences*. Includes a noisy-hardware tomography that reconstructs a *mixed* state.

**Architecture:** A `qot_course.quantum.density` module of pure functions on density matrices. Visualization helper `plot_density_matrix` (Re/Im heatmaps) added to `qot_course.viz`; the Bloch ball reuses `viz.plot_bloch` with a sub-unit Bloch vector. The notebook composes them and runs single-qubit tomography on a noisy `AerSimulator`.

**Tech Stack:** numpy, scipy (`linalg.sqrtm` for fidelity), matplotlib, qiskit/qiskit-aer (noisy backend), existing `qot_course.viz` and `qot_course.quantum.states`.

**Plan series:** Plan 4. Applies the `notebooks-must-be-didactic` standard (rich, every figure explained).

---

## File structure

```
src/qot_course/quantum/density.py       # density ops
src/qot_course/viz.py                    # + plot_density_matrix()
tests/test_density.py                    # TDD
tests/test_viz_density.py                # TDD for the viz helper
notebooks/s03_density.ipynb              # the session
summaries/build_s03_summary.py           # one-page PDF
docs/dictionary.tex                      # + von Neumann entropy row
```

---

### Task 1: `quantum.density` module

**Files:**
- Create: `src/qot_course/quantum/density.py`
- Test: `tests/test_density.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_density.py
import numpy as np
import pytest

from qot_course.quantum.states import KET_0, KET_1, KET_PLUS
from qot_course.quantum.density import (
    density_matrix,
    maximally_mixed,
    is_density_matrix,
    purity,
    von_neumann_entropy,
    fidelity,
    trace_distance,
    bloch_vector,
    density_from_bloch,
)


def test_pure_state_density_is_valid_and_pure():
    rho = density_matrix(KET_PLUS)
    assert is_density_matrix(rho)
    assert purity(rho) == pytest.approx(1.0)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-9)


def test_maximally_mixed_has_half_purity_and_one_bit_entropy():
    rho = maximally_mixed(2)
    assert purity(rho) == pytest.approx(0.5)
    assert von_neumann_entropy(rho) == pytest.approx(1.0)


def test_plus_and_mixed_share_diagonal_but_differ_offdiagonal():
    plus = density_matrix(KET_PLUS)
    mixed = maximally_mixed(2)
    np.testing.assert_allclose(np.diag(plus), np.diag(mixed), atol=1e-12)  # same Z-statistics
    assert not np.allclose(plus, mixed)  # but different states (coherences)


def test_fidelity_and_trace_distance_extremes():
    r0, r1 = density_matrix(KET_0), density_matrix(KET_1)
    assert fidelity(r0, r0) == pytest.approx(1.0, abs=1e-9)
    assert fidelity(r0, r1) == pytest.approx(0.0, abs=1e-9)
    assert trace_distance(r0, r1) == pytest.approx(1.0, abs=1e-9)
    assert trace_distance(r0, r0) == pytest.approx(0.0, abs=1e-9)


def test_bloch_roundtrip():
    rho = density_matrix(KET_PLUS)
    np.testing.assert_allclose(density_from_bloch(bloch_vector(rho)), rho, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_density.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.quantum.density'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/quantum/density.py
"""Density matrices: construction, purity, entropy, fidelity, trace distance."""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Return the pure-state density matrix rho = |psi><psi|."""
    psi = np.asarray(state, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def maximally_mixed(dim: int = 2) -> np.ndarray:
    """Return the maximally mixed state I / dim."""
    return np.eye(dim, dtype=complex) / dim


def mixed_state(states: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Return the statistical mixture sum_i w_i |psi_i><psi_i| (weights need not be normalised)."""
    rho = sum(w * density_matrix(s) for s, w in zip(states, weights))
    return rho / np.trace(rho).real


def _symmetrise(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return (rho + rho.conj().T) / 2


def is_density_matrix(rho: np.ndarray, atol: float = 1e-9) -> bool:
    """Check Hermitian, unit-trace, positive-semidefinite."""
    rho = np.asarray(rho, dtype=complex)
    hermitian = np.allclose(rho, rho.conj().T, atol=atol)
    unit_trace = abs(np.trace(rho) - 1.0) < atol
    psd = bool(np.all(np.linalg.eigvalsh(_symmetrise(rho)) > -atol))
    return bool(hermitian and unit_trace and psd)


def purity(rho: np.ndarray) -> float:
    """Return tr(rho^2) in [1/dim, 1]; equals 1 iff pure."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """Return S(rho) = -tr(rho log rho), in bits by default."""
    vals = np.linalg.eigvalsh(_symmetrise(rho))
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log(vals)) / np.log(base))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return float(np.real(np.trace(inner)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return T(rho, sigma) = 1/2 ||rho - sigma||_1."""
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    vals = np.linalg.eigvalsh(_symmetrise(diff))
    return float(0.5 * np.sum(np.abs(vals)))


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Bloch vector r = (tr(rho X), tr(rho Y), tr(rho Z)) of a qubit density matrix."""
    rho = np.asarray(rho, dtype=complex)
    return np.array([float(np.real(np.trace(rho @ M))) for M in (_X, _Y, _Z)])


def density_from_bloch(r: np.ndarray) -> np.ndarray:
    """Qubit density matrix rho = 1/2 (I + r_x X + r_y Y + r_z Z)."""
    r = np.asarray(r, dtype=float)
    return 0.5 * (np.eye(2, dtype=complex) + r[0] * _X + r[1] * _Y + r[2] * _Z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_density.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/quantum/density.py tests/test_density.py
git commit -m "feat(quantum): add density matrices, purity, entropy, fidelity"
```

---

### Task 2: `plot_density_matrix` visualization

**Files:**
- Modify: `src/qot_course/viz.py`
- Test: `tests/test_viz_density.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viz_density.py
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qot_course import viz
from qot_course.quantum.density import density_matrix
from qot_course.quantum.states import KET_PLUS


def test_plot_density_matrix_returns_figure():
    rho = density_matrix(KET_PLUS)
    assert isinstance(viz.plot_density_matrix(rho, title="|+>"), Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_viz_density.py -v`
Expected: FAIL with `AttributeError: module 'qot_course.viz' has no attribute 'plot_density_matrix'`

- [ ] **Step 3: Append to `src/qot_course/viz.py`**

```python
def plot_density_matrix(rho, title: str = "") -> plt.Figure:
    """Show the real and imaginary parts of a density matrix as annotated heatmaps."""
    import numpy as np

    rho = np.asarray(rho, dtype=complex)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, part, name in zip(axes, (rho.real, rho.imag), ("Re(ρ)", "Im(ρ)")):
        im = ax.imshow(part, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(name, pad=10)
        ax.set_xticks(range(part.shape[1]))
        ax.set_yticks(range(part.shape[0]))
        ax.grid(False)
        for i in range(part.shape[0]):
            for j in range(part.shape[1]):
                ax.annotate(f"{part[i, j]:.2f}", (j, i), ha="center", va="center",
                            color="#0d1117", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_viz_density.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/viz.py tests/test_viz_density.py
git commit -m "feat(viz): add density-matrix Re/Im heatmap plot"
```

---

### Task 3: Didactic notebook `s03_density.ipynb`

**Files:**
- Create: `notebooks/s03_density.ipynb`

- [ ] **Step 1: Build the notebook (every figure followed by a "read the figure" cell)**

Section plan:
1. **Title block** + Purpose.
2. **§0 Objectives.**
3. **Setup** (code): seeds, env, imports, `viz.use_course_style()`.
4. **§1 From a state to a density matrix** (md): ρ = |ψ><ψ|; why operators.
5. (code) `rho_plus = density_matrix(KET_PLUS)`; `viz.plot_density_matrix(rho_plus, "ρ for |+>")`.
6. (md) read the figure: diagonal = Z-probabilities; **off-diagonal = coherence**.
7. **§2 Pure vs mixed** (md): classical uncertainty → statistical mixture; the maximally mixed state.
8. (code) `mixed = maximally_mixed(2)`; `viz.plot_density_matrix(mixed, "ρ for I/2")`.
9. (md) **The key lesson**: `|+>` and `I/2` have the *same diagonal* (same 50/50 Z-statistics) but `|+>` has off-diagonal coherences and `I/2` does not. A Z-measurement cannot tell them apart — the difference is purely quantum.
10. (code) print `born_probabilities`-equivalent diagonals; `purity` and `von_neumann_entropy` for both (1 & 0 bits vs 0.5 & 1 bit).
11. **§3 The Bloch ball** (md): pure on the surface, mixed inside, I/2 at the centre.
12. (code) `viz.plot_bloch(KET_PLUS, "pure |+> (surface)")`; reconstruct a mixed Bloch vector `density.bloch_vector(0.5*rho_plus + 0.5*mixed)` and plot it (inside the ball).
13. (md) read the figures.
14. **§4 Telling states apart** (md): fidelity and trace distance.
15. (code) a small table of `fidelity`/`trace_distance` between `|0>`, `|+>`, `I/2`.
16. (md) read the table.
17. **§5 Tomography: a real (noisy) state is mixed** (md): estimate <X>,<Y>,<Z> of an H-prepared qubit on a *noisy* backend, reconstruct ρ.
18. (code) use `qot_course.hardware.runtime.get_noisy_backend()`; for each basis rotate+measure, estimate expectation; `r = [<X>,<Y>,<Z>]`; `rho_hat = density.density_from_bloch(r)`; print `purity(rho_hat)` (< 1) and `viz.plot_density_matrix(rho_hat)`.
19. (md) read the figure: purity < 1, the Bloch vector is *inside* the sphere — the prepared state is **mixed** because the device is noisy. This is why ρ, not state vectors, is the right object.
20. **§6 Dictionary update** (md): Shannon entropy ↔ von Neumann entropy.
21. **§7 Exercises.**
22. **Conclusions & references** (Nielsen & Chuang ch. 2, 8; Wilde *Quantum Information Theory*).

- [ ] **Step 2: Execute end-to-end**

Run: `uv --directory . run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=240 notebooks/s03_density.ipynb`
Expected: exit 0.

- [ ] **Step 3: Clear outputs**

Run: `uv --directory . run jupyter nbconvert --clear-output --inplace notebooks/s03_density.ipynb`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add notebooks/s03_density.ipynb
git commit -m "feat(s3): add didactic density-matrices notebook"
```

---

### Task 4: Summary PDF

**Files:**
- Create: `summaries/build_s03_summary.py`

- [ ] **Step 1: Write the builder**

```python
# summaries/build_s03_summary.py
"""Build the Session 3 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Density matrices --- the central object}

\textbf{Definition.} A density matrix $\rho$ is Hermitian, positive-semidefinite, with
$\mathrm{tr}\,\rho = 1$. Pure states are $\rho = |\psi\rangle\langle\psi|$
($\mathrm{tr}\,\rho^2 = 1$); mixed states have $\mathrm{tr}\,\rho^2 < 1$.

\textbf{Coherence.} The diagonal of $\rho$ gives the measurement probabilities; the
\emph{off-diagonal} entries encode quantum coherence. $|+\rangle\langle+|$ and the
maximally mixed $I/2$ share a diagonal yet differ off-diagonal --- a Z-measurement cannot
tell them apart.

\textbf{Measures.} von Neumann entropy $S(\rho) = -\mathrm{tr}(\rho\log\rho)$ (0 for pure,
1 bit for $I/2$); Uhlmann fidelity and trace distance compare states. Noisy hardware
reconstructs a \emph{mixed} $\rho$ (purity $< 1$).
"""


def main() -> Path:
    """Build the S3 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 3 --- Density Matrices",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s03_summary",
    )


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 2: Run it**

Run: `uv --directory . run python summaries/build_s03_summary.py`
Expected: prints a path ending `summaries/s03_summary.pdf`; the file exists.

- [ ] **Step 3: Commit**

```bash
git add summaries/build_s03_summary.py
git commit -m "feat(s3): add session 3 summary PDF generator"
```

---

### Task 5: Dictionary row

**Files:**
- Modify: `docs/dictionary.tex`

- [ ] **Step 1: Add after the Born-rule row**, before the `% one row added per session` comment:

```latex
Shannon entropy $H(p)$ & von Neumann entropy $S(\rho) = -\mathrm{tr}(\rho\log\rho)$ \\
```

- [ ] **Step 2: Verify it compiles**

Run: `command -v latexmk && (cd docs && latexmk -pdf -interaction=nonstopmode dictionary.tex) || echo "skip"`
Expected: `dictionary.pdf` produced.

- [ ] **Step 3: Commit**

```bash
git add docs/dictionary.tex
git commit -m "docs(s3): add von Neumann entropy row to the dictionary"
```

---

## Self-Review

**1. Spec coverage (S3 row).** ρ as PSD/trace-1 (Task 1), pure vs mixed + purity + von Neumann entropy (Task 1, notebook), fidelity & trace distance (Task 1), the diagonal-collapse/coherence lesson (`|+>` vs `I/2`, Task 1 test + notebook §2), the Bloch ball (notebook §3), and noisy-hardware tomography reconstructing a mixed state (notebook §5). Deliverable contract (module/viz/notebook/summary/dictionary) covered.

**2. Placeholder scan.** Complete runnable code in module/viz/summary steps; the notebook task is a concrete section plan with exact helper calls; full markdown written during execution to the didactic standard. No TBD/TODO.

**3. Type/name consistency.** `density_matrix`, `maximally_mixed`, `purity`, `von_neumann_entropy`, `fidelity`, `trace_distance`, `bloch_vector`, `density_from_bloch` defined in Task 1 and used consistently in Tasks 2–3. `viz.plot_density_matrix(rho, title="")` defined in Task 2, used in Task 3. Note `density.bloch_vector(rho)` (density-matrix input) is distinct from `states.bloch_vector(state)` (state-vector input) — used module-qualified. `get_noisy_backend()` matches the Sprint-0 `hardware.runtime` definition.

---

*Next plan: **S4 — Composite systems & channels** (tensor product, partial trace, entanglement, CPTP/Kraus; a Bell state on hardware).*
