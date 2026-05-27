# S1 — Teaser & Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Session 1 — a light `intro` module (environment check + a 1D transport visual), an executable roadmap notebook with the first real `ot.emd2`, and a one-page syllabus PDF.

**Architecture:** A small `qot_course.intro` module (tested), consumed by `notebooks/s01_roadmap.ipynb`; the syllabus PDF is produced by a thin script calling the Sprint-0 `build_summary` pipeline. No new dependencies.

**Tech Stack:** numpy, matplotlib, POT (`ot`), the existing `qot_course.summaries.build`, Jupyter (`nbconvert`), latexmk.

**Plan series:** Plan 2 of the course (after Sprint 0 foundations). Builds on `qot_course.summaries.build` and the installed stack (qiskit 2.4.1 / aer 0.17.2 / runtime 0.47.0).

---

## File structure (created by this plan)

```
src/qot_course/intro.py                 # check_environment() + plot_1d_transport()
tests/test_intro.py                     # TDD for the module
notebooks/s01_roadmap.ipynb             # the executable roadmap session
summaries/build_s01_syllabus.py         # builds summaries/s01_syllabus.pdf via build_summary
```

---

### Task 1: `intro` module (environment check + 1D transport plot)

**Files:**
- Create: `src/qot_course/intro.py`
- Test: `tests/test_intro.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intro.py
import matplotlib

matplotlib.use("Agg")  # headless backend; set before pyplot is imported via intro

import numpy as np
from matplotlib.figure import Figure

from qot_course.intro import check_environment, plot_1d_transport


def test_check_environment_reports_core_packages():
    env = check_environment()
    assert env["numpy"] is not None  # numpy is installed
    assert "qiskit" in env  # key reported even if value could be a version string


def test_plot_1d_transport_returns_figure():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.5, 0.5])
    fig = plot_1d_transport(source, target)
    assert isinstance(fig, Figure)


def test_plot_1d_transport_with_plan_runs():
    source = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])
    plan = np.array([[0.0, 1.0], [0.0, 0.0]])
    fig = plot_1d_transport(source, target, transport_plan=plan)
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_intro.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.intro'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/intro.py
"""Session 1 helpers: environment check and a 1D optimal-transport visual."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import matplotlib.pyplot as plt
import numpy as np

_KEY_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pot",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "cvxpy",
]


def check_environment() -> dict[str, str | None]:
    """Return installed versions of the course's key packages.

    Maps each package name to its version string, or ``None`` if missing.
    """
    versions: dict[str, str | None] = {}
    for package in _KEY_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def plot_1d_transport(
    source: np.ndarray,
    target: np.ndarray,
    transport_plan: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot two 1D mass distributions and, optionally, a transport plan as arrows.

    Parameters
    ----------
    source, target : np.ndarray
        1D mass vectors over integer positions ``0..n-1``.
    transport_plan : np.ndarray, optional
        ``(n, n)`` plan where ``P[i, j]`` is the mass moved from ``i`` to ``j``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    x = np.arange(len(source))
    ax.bar(x - 0.15, source, width=0.3, label="source", alpha=0.8)
    ax.bar(x + 0.15, target, width=0.3, label="target", alpha=0.8)

    if transport_plan is not None:
        plan = np.asarray(transport_plan, dtype=float)
        peak = plan.max() if plan.max() > 0 else 1.0
        for i in range(plan.shape[0]):
            for j in range(plan.shape[1]):
                if plan[i, j] > 1e-9:
                    ax.annotate(
                        "",
                        xy=(j + 0.15, target[j]),
                        xytext=(i - 0.15, source[i]),
                        arrowprops={"arrowstyle": "->", "alpha": 0.2 + 0.6 * plan[i, j] / peak},
                    )

    ax.set_xlabel("position")
    ax.set_ylabel("mass")
    ax.legend()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_intro.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/intro.py tests/test_intro.py
git commit -m "feat(intro): add environment check and 1D transport plot"
```

---

### Task 2: Roadmap notebook

**Files:**
- Create: `notebooks/s01_roadmap.ipynb`

- [ ] **Step 1: Create the notebook with these cells (in order)**

Cell 1 — markdown:
```markdown
# Session 1 — Teaser & Roadmap

Welcome. By the end of this course you will compute a genuine *quantum* Wasserstein
distance. Today: see the destination, run your first optimal transport, and meet the
classical ↔ quantum dictionary we will grow every week.
```

Cell 2 — code:
```python
import numpy as np
import matplotlib.pyplot as plt
import ot

from qot_course.intro import check_environment, plot_1d_transport

# 1. Does everyone's environment work?
for package, version in check_environment().items():
    print(f"{package:22s} {version or 'MISSING'}")
```

Cell 3 — markdown:
```markdown
## Optimal transport in one picture

A pile of mass (`source`) must be reshaped into another pile (`target`) at least cost.
The cost of moving a unit from position `i` to `j` is the squared distance `(i - j)^2`.
`POT` solves the exact problem and returns the optimal **transport plan**.
```

Cell 4 — code:
```python
source = np.array([0.0, 0.1, 0.6, 0.3, 0.0])
target = np.array([0.2, 0.4, 0.0, 0.1, 0.3])
source = source / source.sum()
target = target / target.sum()

positions = np.arange(len(source)).reshape(-1, 1)
cost = ot.dist(positions, positions, metric="sqeuclidean")

plan = ot.emd(source, target, cost)          # exact optimal transport plan
w2_squared = float(np.sum(plan * cost))       # squared 2-Wasserstein distance
print(f"W2^2 = {w2_squared:.4f}")

fig = plot_1d_transport(source, target, transport_plan=plan)
fig.suptitle(f"Optimal transport plan  (W2^2 = {w2_squared:.3f})")
plt.show()
```

Cell 5 — markdown:
```markdown
## The destination: quantizing this

Replace probability vectors by **density matrices** and you get *quantum* optimal
transport. We will build the dictionary that bridges the two, one row per session:

| Classical | Quantum |
|-----------|---------|
| probability vector `p` | density matrix `ρ` (diagonal ⇒ classical) |
| marginal | partial trace |

## Roadmap (four movements)

1. **Quantum foundations** — qubits, density matrices, channels.
2. **Information theory & geometry** — entropy, KL, mutual information; Fisher–Rao.
3. **Classical optimal transport** — Monge–Kantorovich, Wasserstein, Sinkhorn.
4. **Quantum optimal transport** — the SDP, quantum Sinkhorn, a research capstone.

**Next session (S2):** qubits & states.
```

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs**

Run: `uv --directory . run jupyter nbconvert --to notebook --execute --inplace notebooks/s01_roadmap.ipynb`
Expected: exit code 0, no cell errors.

- [ ] **Step 3: Clear outputs before committing (notebook hygiene)**

Run: `uv --directory . run jupyter nbconvert --clear-output --inplace notebooks/s01_roadmap.ipynb`
Expected: exit code 0; the notebook has no stored outputs.

- [ ] **Step 4: Commit**

```bash
git add notebooks/s01_roadmap.ipynb
git commit -m "feat(s1): add roadmap notebook with first optimal transport"
```

---

### Task 3: Syllabus PDF generator

**Files:**
- Create: `summaries/build_s01_syllabus.py`

- [ ] **Step 1: Write the syllabus builder**

```python
# summaries/build_s01_syllabus.py
"""Build the Session 1 syllabus PDF (the course roadmap) via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{A 16-session journey: classical optimal transport to its quantum cousin}

\textbf{M1 --- Quantum foundations:} qubits; density matrices; composite systems and channels.

\textbf{M2 --- Information theory \& geometry:} entropy, KL divergence, mutual information,
transfer entropy; Fisher--Rao geometry; quantum information.

\textbf{M3 --- Classical optimal transport:} Monge--Kantorovich; Wasserstein distances;
duality and Sinkhorn; Gaussians and the dynamic formulation.

\textbf{M4 --- Quantum optimal transport:} why QOT; the coupling semidefinite program;
quantum Sinkhorn; a hyperscanning capstone; the research frontier.

\medskip
Each session ships a code module, an exhaustive notebook, and a one-page summary like this one.
"""


def main() -> Path:
    """Build the syllabus PDF and return its path."""
    return build_summary(
        {
            "title": r"Quantum Optimal Transport --- Course Roadmap",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s01_syllabus",
    )


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 2: Run it to verify the PDF builds**

Run: `uv --directory . run python summaries/build_s01_syllabus.py`
Expected: prints a path ending in `summaries/s01_syllabus.pdf`, and that file exists (it is git-ignored).

- [ ] **Step 3: Commit (script only; the PDF is git-ignored)**

```bash
git add summaries/build_s01_syllabus.py
git commit -m "feat(s1): add syllabus PDF generator"
```

---

## Self-Review

**1. Spec coverage (S1 row).** The spec's S1 deliverables are met: the roadmap notebook with the first `ot.emd2` (Task 2), the `env_check` (Task 1, `check_environment`), the syllabus PDF via `build_summary` (Task 3), and the dictionary preview (shown in the notebook; the `docs/dictionary.tex` skeleton already holds the two seed rows from Sprint 0, so no new rows are due until S2).

**2. Placeholder scan.** No TBD/TODO; every code step is complete and runnable; commands have expected output.

**3. Type/name consistency.** `check_environment() -> dict[str, str | None]` and `plot_1d_transport(source, target, transport_plan=None, ax=None) -> Figure` are defined in Task 1 and used with matching signatures in the Task 2 notebook. `build_summary(context, out_dir, stem, template="summary.tex.j2")` (defined in Sprint 0) is called with matching keyword/positional arguments in Task 3. The notebook imports only names defined in Task 1 plus `ot` and `numpy`.

---

*Next plan: **S2 — Qubits & states** (the `qot_course.quantum.states` module + notebook + summary; first hardware cell).*
