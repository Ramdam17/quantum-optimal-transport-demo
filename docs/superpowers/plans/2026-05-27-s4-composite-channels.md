# S4 — Composite Systems & Channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Session 4 (closing Movement 1) — tensor products, the **partial trace** (= quantum marginal), **entanglement** (a Bell state whose parts are maximally mixed), and **quantum channels** (CPTP / Kraus), with a `quantum.composite` module, a didactic notebook, a summary, and dictionary rows.

**Architecture:** A `qot_course.quantum.composite` module: `tensor`, `partial_trace` (einsum-based, multi-subsystem), `bell_state`, `entanglement_entropy`, `apply_channel`, `depolarizing_channel`, `is_cptp`. Reuses `density.von_neumann_entropy` and `viz.plot_density_matrix` (no new viz). The notebook builds two-qubit states and shows noise as a channel.

**Tech Stack:** numpy, existing `qot_course.quantum.density` and `qot_course.viz`.

**Plan series:** Plan 5 (last of Movement 1). Applies the `notebooks-must-be-didactic` standard.

---

## File structure

```
src/qot_course/quantum/composite.py     # tensor, partial_trace, bell, entanglement, channels
tests/test_composite.py                  # TDD
notebooks/s04_composite.ipynb            # the session
summaries/build_s04_summary.py           # one-page PDF
docs/dictionary.tex                      # + channel & entanglement rows
```

---

### Task 1: `quantum.composite` module

**Files:**
- Create: `src/qot_course/quantum/composite.py`
- Test: `tests/test_composite.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composite.py
import numpy as np
import pytest

from qot_course.quantum.states import KET_0, KET_PLUS
from qot_course.quantum.density import density_matrix, maximally_mixed, purity
from qot_course.quantum.composite import (
    tensor,
    partial_trace,
    bell_state,
    entanglement_entropy,
    apply_channel,
    depolarizing_channel,
    is_cptp,
)


def test_tensor_dimension():
    assert tensor(KET_0, KET_0).shape == (4,)


def test_partial_trace_of_product_recovers_factor():
    rho = density_matrix(tensor(KET_0, KET_PLUS))
    reduced_A = partial_trace(rho, keep=[0], dims=[2, 2])
    np.testing.assert_allclose(reduced_A, density_matrix(KET_0), atol=1e-12)


def test_bell_state_parts_are_maximally_mixed():
    rho = density_matrix(bell_state())
    reduced_A = partial_trace(rho, keep=[0], dims=[2, 2])
    np.testing.assert_allclose(reduced_A, maximally_mixed(2), atol=1e-12)


def test_bell_state_has_one_bit_of_entanglement():
    rho = density_matrix(bell_state())
    assert entanglement_entropy(rho, dims=[2, 2]) == pytest.approx(1.0)


def test_product_state_has_no_entanglement():
    rho = density_matrix(tensor(KET_0, KET_PLUS))
    assert entanglement_entropy(rho, dims=[2, 2]) == pytest.approx(0.0, abs=1e-9)


def test_depolarizing_channel_is_cptp_and_shrinks_purity():
    kraus = depolarizing_channel(0.5)
    assert is_cptp(kraus)
    rho = density_matrix(KET_PLUS)
    out = apply_channel(rho, kraus)
    assert purity(out) < purity(rho)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_composite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.quantum.composite'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/quantum/composite.py
"""Composite quantum systems: tensor products, partial trace, entanglement, channels."""

from __future__ import annotations

import numpy as np

from qot_course.quantum.density import von_neumann_entropy

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def tensor(*operands: np.ndarray) -> np.ndarray:
    """Kronecker (tensor) product of states or operators, left to right."""
    result = np.asarray(operands[0], dtype=complex)
    for op in operands[1:]:
        result = np.kron(result, np.asarray(op, dtype=complex))
    return result


def bell_state() -> np.ndarray:
    """Return the Bell state (|00> + |11>) / sqrt(2)."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1.0
    return psi / np.sqrt(2)


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    """Trace out every subsystem not in ``keep``.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix on the composite system.
    keep : list[int]
        Indices of the subsystems to keep.
    dims : list[int]
        Dimension of each subsystem.
    """
    keep = sorted(keep)
    n = len(dims)
    rho = np.asarray(rho, dtype=complex).reshape(dims + dims)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for i in range(n):
        if i not in keep:  # contract row with col -> trace this subsystem out
            col[i] = row[i]
    out = [row[i] for i in keep] + [col[i] for i in keep]
    reduced = np.einsum(rho, row + col, out)
    d_keep = int(np.prod([dims[i] for i in keep]))
    return reduced.reshape(d_keep, d_keep)


def entanglement_entropy(rho: np.ndarray, dims: list[int]) -> float:
    """Entanglement entropy = von Neumann entropy of the first subsystem's reduced state."""
    reduced = partial_trace(rho, keep=[0], dims=dims)
    return von_neumann_entropy(reduced)


def apply_channel(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    """Apply a quantum channel rho -> sum_k K_k rho K_k^dagger."""
    rho = np.asarray(rho, dtype=complex)
    return sum(K @ rho @ K.conj().T for K in kraus)


def is_cptp(kraus: list[np.ndarray], atol: float = 1e-9) -> bool:
    """Check the completeness relation sum_k K_k^dagger K_k = I (trace preservation)."""
    dim = kraus[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus)
    return bool(np.allclose(total, np.eye(dim, dtype=complex), atol=atol))


def depolarizing_channel(p: float) -> list[np.ndarray]:
    """Kraus operators of the single-qubit depolarizing channel with parameter ``p``."""
    return [
        np.sqrt(1.0 - 3.0 * p / 4.0) * _I,
        np.sqrt(p / 4.0) * _X,
        np.sqrt(p / 4.0) * _Y,
        np.sqrt(p / 4.0) * _Z,
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_composite.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/quantum/composite.py tests/test_composite.py
git commit -m "feat(quantum): add tensor, partial trace, entanglement, channels"
```

---

### Task 2: Didactic notebook `s04_composite.ipynb`

**Files:**
- Create: `notebooks/s04_composite.ipynb`

- [ ] **Step 1: Build the notebook (every figure followed by a "read the figure" cell)**

Section plan:
1. **Title block** + Purpose.
2. **§0 Objectives.**
3. **Setup** (code): seeds, imports, `viz.use_course_style()`.
4. **§1 Two systems: the tensor product** (md): combining spaces; `tensor`.
5. (code) build `|0>⊗|+>`; its 4×4 density matrix; `viz.plot_density_matrix`.
6. (md) read the figure.
7. **§2 The partial trace = the quantum marginal** (md): callback to the classical marginal.
8. (code) `partial_trace` of the product state → recover `|0>` and `|+>`; print/plot.
9. (md) read: tracing out B recovers A exactly, just like a classical marginal.
10. **§3 Entanglement** (md): the Bell state; the whole is pure but the parts are not.
11. (code) `bell_state`; global purity = 1; `partial_trace` → `I/2`; `entanglement_entropy` = 1 bit; plot the Bell ρ and the reduced ρ side by side.
12. (md) **the punchline**: a pure global state with maximally mixed parts — the parts are *less* defined than the whole. No classical joint distribution does this. This is entanglement.
13. **§4 Quantum channels = noisy evolution** (md): CPTP / Kraus; the depolarizing channel; callback "channel = quantum Markov kernel".
14. (code) `depolarizing_channel(p)`; `is_cptp`; apply to `|+>` for increasing `p`; plot purity vs p (mass shrinks toward I/2).
15. (md) read the figure: more noise → more mixed → Bloch vector shrinks; this is exactly the noise that made S3's tomography mixed.
16. **§5 Dictionary update** (md): two new rows (channel, entanglement/independence).
17. **§6 Exercises.**
18. **Conclusions & references** (Nielsen & Chuang ch. 2, 8; Wilde ch. 4).

- [ ] **Step 2: Execute end-to-end**

Run: `uv --directory . run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/s04_composite.ipynb`
Expected: exit 0.

- [ ] **Step 3: Clear outputs**

Run: `uv --directory . run jupyter nbconvert --clear-output --inplace notebooks/s04_composite.ipynb`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add notebooks/s04_composite.ipynb
git commit -m "feat(s4): add didactic composite-systems & channels notebook"
```

---

### Task 3: Summary PDF

**Files:**
- Create: `summaries/build_s04_summary.py`

- [ ] **Step 1: Write the builder**

```python
# summaries/build_s04_summary.py
"""Build the Session 4 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Composite systems and channels}

\textbf{Tensor product.} Two systems combine as $\mathcal{H}_A \otimes \mathcal{H}_B$;
states and operators combine by the Kronecker product.

\textbf{Partial trace} is the quantum marginal: $\rho_A = \mathrm{tr}_B\,\rho_{AB}$ recovers
the state of $A$ alone.

\textbf{Entanglement.} The Bell state $(|00\rangle+|11\rangle)/\sqrt2$ is \emph{pure} yet
its parts are \emph{maximally mixed} ($\rho_A = I/2$, entanglement entropy $= 1$ bit): the
whole is more defined than its parts --- impossible classically.

\textbf{Quantum channels} (the quantum Markov kernels) are completely-positive
trace-preserving maps, $\rho \mapsto \sum_k K_k \rho K_k^\dagger$ with
$\sum_k K_k^\dagger K_k = I$. The depolarizing channel mixes a state toward $I/2$ --- the
noise behind S3's tomography.
"""


def main() -> Path:
    """Build the S4 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 4 --- Composite Systems & Channels",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s04_summary",
    )


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 2: Run it**

Run: `uv --directory . run python summaries/build_s04_summary.py`
Expected: prints a path ending `summaries/s04_summary.pdf`; the file exists.

- [ ] **Step 3: Commit**

```bash
git add summaries/build_s04_summary.py
git commit -m "feat(s4): add session 4 summary PDF generator"
```

---

### Task 4: Dictionary rows

**Files:**
- Modify: `docs/dictionary.tex`

- [ ] **Step 1: Add two rows** after the von Neumann entropy row, before the `% one row added per session` comment:

```latex
Markov kernel & quantum channel (CPTP map) \\
independent variables & product state (no entanglement) \\
```

- [ ] **Step 2: Verify it compiles**

Run: `command -v latexmk && (cd docs && latexmk -pdf -interaction=nonstopmode dictionary.tex) || echo "skip"`
Expected: `dictionary.pdf` produced.

- [ ] **Step 3: Commit**

```bash
git add docs/dictionary.tex
git commit -m "docs(s4): add channel and entanglement rows to the dictionary"
```

---

## Self-Review

**1. Spec coverage (S4 row).** Tensor product, partial trace, entanglement (Tasks 1 & notebook §1–3), quantum channels CPTP/Kraus (Task 1 & notebook §4), Bell state on a composite system (Task 1; a hardware Bell-state run is deferred — the noise/channel link is shown via the depolarizing channel which reproduces S3's tomography). Deliverable contract met (module/notebook/summary/dictionary; no new viz — `plot_density_matrix` is reused).

**2. Placeholder scan.** Complete runnable code in module/summary steps; notebook is a concrete section plan with exact helper calls; full markdown written during execution. No TBD/TODO.

**3. Type/name consistency.** `tensor`, `partial_trace(rho, keep, dims)`, `bell_state`, `entanglement_entropy(rho, dims)`, `apply_channel(rho, kraus)`, `depolarizing_channel(p)`, `is_cptp(kraus)` defined in Task 1, used consistently in Task 2. Reuses `density.von_neumann_entropy`, `density.purity`, `density.maximally_mixed`, `viz.plot_density_matrix` with their established signatures.

---

*Movement 1 (Quantum foundations) complete after this plan. Next: Movement 2 — **S5 Classical information theory** (entropy, KL, mutual information, transfer entropy).*
