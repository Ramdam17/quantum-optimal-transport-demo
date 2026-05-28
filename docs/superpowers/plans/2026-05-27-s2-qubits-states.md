# S2 — Qubits & States Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Session 2 — build the qubit from zero (amplitudes, Bloch sphere, Born rule, measurement & shot noise), with a `quantum.states` module, Bloch/measurement visualizations, a rich didactic notebook (every figure explained), a summary PDF, and a new dictionary row.

**Architecture:** A `qot_course.quantum.states` module of small pure functions (state construction, Bloch coordinates, Born-rule probabilities, measurement sampling). Visualization helpers added to `qot_course.viz` (Bloch sphere via Qiskit; styled measurement histogram). The notebook composes them; a Qiskit `AerSimulator` cell shows shot noise (simulator-first, hardware optional).

**Tech Stack:** numpy, matplotlib, qiskit (`plot_bloch_vector`, `QuantumCircuit`, `qiskit_aer.AerSimulator`), the existing `qot_course.viz`.

**Plan series:** Plan 3 of the course. **Applies the didactic standard from the S1 redo:** rich progressive notebook, every figure explained, reusable plotting in `viz`.

---

## File structure (created/modified by this plan)

```
src/qot_course/quantum/__init__.py      # new subpackage
src/qot_course/quantum/states.py        # qubit_state, bloch_vector, born_probabilities, sample_counts
src/qot_course/viz.py                   # + plot_bloch(), plot_counts()
tests/test_states.py                    # TDD for the module
tests/test_viz_quantum.py               # TDD for the two new viz helpers
notebooks/s02_qubits.ipynb              # the didactic session
summaries/build_s02_summary.py          # one-page PDF
docs/dictionary.tex                     # + Born-rule row
```

---

### Task 1: `quantum.states` module

**Files:**
- Create: `src/qot_course/quantum/__init__.py` (empty), `src/qot_course/quantum/states.py`
- Test: `tests/test_states.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_states.py
import numpy as np
import pytest

from qot_course.quantum.states import (
    KET_0,
    KET_PLUS,
    qubit_state,
    bloch_vector,
    born_probabilities,
    sample_counts,
)


def test_qubit_state_basis():
    np.testing.assert_allclose(qubit_state(0.0), KET_0, atol=1e-12)


def test_bloch_vector_of_ket0_points_up():
    np.testing.assert_allclose(bloch_vector(KET_0), [0.0, 0.0, 1.0], atol=1e-12)


def test_bloch_vector_of_plus_points_along_x():
    np.testing.assert_allclose(bloch_vector(KET_PLUS), [1.0, 0.0, 0.0], atol=1e-12)


def test_born_probabilities_of_plus_are_uniform():
    probs = born_probabilities(KET_PLUS)
    assert probs["0"] == pytest.approx(0.5)
    assert probs["1"] == pytest.approx(0.5)


def test_sample_counts_sum_to_shots_and_are_reproducible():
    a = sample_counts(KET_PLUS, shots=2048, seed=1)
    b = sample_counts(KET_PLUS, shots=2048, seed=1)
    assert sum(a.values()) == 2048
    assert a == b  # same seed -> same counts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_states.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.quantum'`

- [ ] **Step 3: Write minimal implementation**

`src/qot_course/quantum/__init__.py`:
```python
```

`src/qot_course/quantum/states.py`:
```python
"""Single-qubit states: construction, Bloch coordinates, and the Born rule."""

from __future__ import annotations

import numpy as np

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)
KET_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
KET_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)

_PAULI = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def qubit_state(theta: float, phi: float = 0.0) -> np.ndarray:
    """Return the pure state cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex
    )


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Return Bloch coordinates (x, y, z) = (<X>, <Y>, <Z>) of a pure qubit state."""
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)
    return np.array(
        [float(np.real(state.conj() @ _PAULI[p] @ state)) for p in ("X", "Y", "Z")]
    )


def born_probabilities(state: np.ndarray) -> dict[str, float]:
    """Born rule in the computational (Z) basis: P(0) = |a0|^2, P(1) = |a1|^2."""
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)
    return {"0": float(abs(state[0]) ** 2), "1": float(abs(state[1]) ** 2)}


def sample_counts(
    state: np.ndarray, shots: int = 1024, seed: int | None = None
) -> dict[str, int]:
    """Simulate ``shots`` computational-basis measurements; return integer counts."""
    probs = born_probabilities(state)
    rng = np.random.default_rng(seed)
    draws = rng.choice(["0", "1"], size=shots, p=[probs["0"], probs["1"]])
    counts = {"0": 0, "1": 0}
    for outcome, n in zip(*np.unique(draws, return_counts=True)):
        counts[str(outcome)] = int(n)
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_states.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/quantum/__init__.py src/qot_course/quantum/states.py tests/test_states.py
git commit -m "feat(quantum): add single-qubit states, Bloch vector, Born rule"
```

---

### Task 2: Bloch + measurement visualizations

**Files:**
- Modify: `src/qot_course/viz.py` (append two functions)
- Test: `tests/test_viz_quantum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viz_quantum.py
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qot_course import viz
from qot_course.quantum.states import KET_PLUS, sample_counts


def test_plot_bloch_returns_figure():
    assert isinstance(viz.plot_bloch(KET_PLUS, title="|+>"), Figure)


def test_plot_counts_returns_figure():
    counts = sample_counts(KET_PLUS, shots=512, seed=0)
    assert isinstance(viz.plot_counts(counts), Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_viz_quantum.py -v`
Expected: FAIL with `AttributeError: module 'qot_course.viz' has no attribute 'plot_bloch'`

- [ ] **Step 3: Append the implementation to `src/qot_course/viz.py`**

```python
def plot_bloch(state, title: str = "") -> plt.Figure:
    """Plot a pure qubit state on the Bloch sphere (via Qiskit)."""
    from qiskit.visualization import plot_bloch_vector

    from qot_course.quantum.states import bloch_vector

    return plot_bloch_vector(list(bloch_vector(state)), title=title)


def plot_counts(counts: dict[str, int], ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart of measurement counts (computational basis)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure
    labels = sorted(counts)
    values = [counts[k] for k in labels]
    ax.bar(labels, values, color=[SOURCE_COLOR, TARGET_COLOR][: len(labels)], alpha=0.9)
    ax.set_xlabel("measurement outcome")
    ax.set_ylabel("counts")
    ax.set_title("Measurement outcomes", pad=12)
    for label, value in zip(labels, values):
        ax.annotate(str(value), (label, value), ha="center", va="bottom", fontsize=11)
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_viz_quantum.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/viz.py tests/test_viz_quantum.py
git commit -m "feat(viz): add Bloch sphere and measurement-counts plots"
```

---

### Task 3: Didactic notebook `s02_qubits.ipynb`

**Files:**
- Create: `notebooks/s02_qubits.ipynb`

- [ ] **Step 1: Build the notebook (sections below; every figure followed by a "read the figure" markdown cell)**

Section plan (markdown + code, narrative-heavy, applying the didactic standard):
1. **Title block** (author/date/project/status + Purpose).
2. **§0 Objectives** — what they'll be able to do.
3. **Setup** (code): seeds, `check_environment`, imports, `viz.use_course_style()`.
4. **§1 From a bit to a qubit** (md): superposition, the 2-vector, normalization.
5. (code) build `KET_0`, `KET_1`, `KET_PLUS`, a general `qubit_state(theta, phi)`; print amplitudes.
6. **§2 Amplitudes & the Born rule** (md): |amplitude|² = probability; measurement is random.
7. (code) `born_probabilities` for `|0>`, `|+>`, and a general state; print.
8. (md) explain the numbers.
9. **§3 The Bloch sphere** (md): geometric picture of a qubit.
10. (code) `viz.plot_bloch(KET_PLUS, title="|+>")` and one for a general state.
11. (md) **read the figure**: poles = |0>/|1>, equator = equal superpositions, the vector = the state.
12. **§4 Measurement & shot noise** (md): each shot is a 0/1 draw; more shots → closer to Born.
13. (code) `sample_counts` at 20, 200, 2000 shots; `viz.plot_counts` for each (or a 1×3 panel).
14. (md) **read the figure**: few shots are noisy, many shots converge to the Born probabilities.
15. **§5 A real circuit with Qiskit** (md): H|0> = |+>; simulate; (optional) hardware.
16. (code) build `QuantumCircuit(1,1)`, `qc.h(0)`, `qc.measure(0,0)`; `qc.draw("mpl")`.
17. (code) run on `AerSimulator()` with shots=2048; `viz.plot_counts(result.get_counts())`.
18. (md) **read the figure** + a note: on real hardware the split is *not* exactly 50/50 (readout error) — the honest reason density matrices (next session) exist. Show the verified hardware idiom in a non-executed markdown block (`QiskitRuntimeService().least_busy(...)`, `SamplerV2`).
19. **§6 Dictionary update** (md): new row — classical probability ↔ Born rule |amplitude|².
20. **§7 Exercises** (md): rotate the state (vary theta/phi) and predict the Bloch vector & counts; measure |+> in the X basis.
21. **Conclusions & References** (md): Nielsen & Chuang ch. 1–2; link to S1.

- [ ] **Step 2: Execute end-to-end**

Run: `uv --directory . run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/s02_qubits.ipynb`
Expected: exit 0, no cell errors.

- [ ] **Step 3: Clear outputs**

Run: `uv --directory . run jupyter nbconvert --clear-output --inplace notebooks/s02_qubits.ipynb`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add notebooks/s02_qubits.ipynb
git commit -m "feat(s2): add didactic qubits & states notebook"
```

---

### Task 4: Summary PDF

**Files:**
- Create: `summaries/build_s02_summary.py`

- [ ] **Step 1: Write the builder**

```python
# summaries/build_s02_summary.py
"""Build the Session 2 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Qubits and states --- the essentials}

\textbf{State.} A qubit is a unit vector $|\psi\rangle = \cos(\theta/2)|0\rangle +
e^{i\phi}\sin(\theta/2)|1\rangle$, a point on the \emph{Bloch sphere}.

\textbf{Born rule.} Measuring in the computational basis gives outcome $x$ with
probability $|\langle x|\psi\rangle|^2$. Measurement is intrinsically random; finite
shots fluctuate around these probabilities (\emph{shot noise}).

\textbf{Why it matters.} Real hardware returns slightly-off probabilities (readout error):
prepared states are never perfectly pure --- the motivation for density matrices (S3).
"""


def main() -> Path:
    """Build the S2 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 2 --- Qubits and States",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s02_summary",
    )


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 2: Run it**

Run: `uv --directory . run python summaries/build_s02_summary.py`
Expected: prints a path ending `summaries/s02_summary.pdf`; the file exists (git-ignored).

- [ ] **Step 3: Commit**

```bash
git add summaries/build_s02_summary.py
git commit -m "feat(s2): add session 2 summary PDF generator"
```

---

### Task 5: Dictionary row

**Files:**
- Modify: `docs/dictionary.tex`

- [ ] **Step 1: Add the Born-rule row** after the `marginal & partial trace` row, before the `% one row added per session` comment:

```latex
event probability $p(x)$ & Born rule $|\langle x|\psi\rangle|^2$ \\
```

- [ ] **Step 2: Verify it still compiles (skip if no latexmk)**

Run: `command -v latexmk && (cd docs && latexmk -pdf -interaction=nonstopmode dictionary.tex) || echo "latexmk absent: skip"`
Expected: `dictionary.pdf` produced (or skip message).

- [ ] **Step 3: Commit**

```bash
git add docs/dictionary.tex
git commit -m "docs(s2): add Born-rule row to the dictionary"
```

---

## Self-Review

**1. Spec coverage (S2 row).** Amplitudes + Bloch sphere (Tasks 1–3), Born rule + measurement/shot-noise (Tasks 1, 3), the optional hardware cell with graceful simulator-first execution and the verified runtime idiom shown (Task 3 §5), the module + notebook + summary + dictionary deliverable contract (Tasks 1–5). Density-matrix motivation is teased (S3 hook), not implemented here.

**2. Placeholder scan.** Module/viz/summary steps contain complete runnable code. The notebook task gives a concrete section-by-section build with the exact helper calls; full markdown is written during execution to the didactic standard (per the `notebooks-must-be-didactic` rule). No TBD/TODO.

**3. Type/name consistency.** `qubit_state`, `bloch_vector`, `born_probabilities`, `sample_counts` are defined in Task 1 and used with matching signatures in Tasks 2–3. `viz.plot_bloch(state, title=...)` and `viz.plot_counts(counts, ax=None)` defined in Task 2, used in Task 3. `build_summary(context, out_dir, stem, ...)` matches its Sprint-0 definition.

---

*Next plan: **S3 — Density matrices** (the central object ρ; pure vs mixed; von Neumann entropy; first real-hardware tomography showing a mixed state).*
