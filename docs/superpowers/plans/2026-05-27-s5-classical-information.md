# S5 — Classical Information Theory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Session 5 (opening Movement 2, the information spine) — Shannon entropy, KL divergence, mutual information, conditional mutual information, and transfer entropy (Schreiber), with an `infotheory.classical` module, a didactic notebook (entropy curve, MI-vs-coupling, TE directionality), a summary, and a dictionary row. Honest caveats on TE estimation and PID.

**Architecture:** A `qot_course.infotheory.classical` module of pure functions on discrete distributions/sequences. Entropies are computed from probability arrays; mutual and conditional-mutual information from joint arrays; transfer entropy reduces to a conditional mutual information of the empirical joint of (target_next, source_past, target_past). Notebook figures are inline (course style), no new viz module.

**Tech Stack:** numpy, matplotlib, existing `qot_course.viz` (style/colors).

**Plan series:** Plan 6 (first of Movement 2). Applies the `notebooks-must-be-didactic` standard.

---

## File structure

```
src/qot_course/infotheory/__init__.py    # new subpackage
src/qot_course/infotheory/classical.py   # entropy, KL, MI, CMI, transfer entropy
tests/test_infotheory.py                  # TDD
notebooks/s05_information.ipynb           # the session
summaries/build_s05_summary.py            # one-page PDF
docs/dictionary.tex                       # + mutual-information row
```

---

### Task 1: `infotheory.classical` module

**Files:**
- Create: `src/qot_course/infotheory/__init__.py` (empty), `src/qot_course/infotheory/classical.py`
- Test: `tests/test_infotheory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infotheory.py
import numpy as np
import pytest

from qot_course.infotheory.classical import (
    shannon_entropy,
    kl_divergence,
    mutual_information,
    conditional_mutual_information,
    transfer_entropy,
)


def test_entropy_of_fair_coin_is_one_bit():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_kl_is_zero_iff_equal_and_nonnegative():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.6, 0.3])
    assert kl_divergence(p, p) == pytest.approx(0.0)
    assert kl_divergence(p, q) > 0.0


def test_mutual_information_independent_is_zero_correlated_is_one():
    independent = np.outer([0.5, 0.5], [0.5, 0.5])
    assert mutual_information(independent) == pytest.approx(0.0, abs=1e-12)
    correlated = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert mutual_information(correlated) == pytest.approx(1.0)


def test_transfer_entropy_is_directional():
    rng = np.random.default_rng(0)
    source = rng.integers(0, 2, size=4000)
    target = np.empty_like(source)
    target[0] = 0
    target[1:] = source[:-1]  # target copies source with lag 1
    te_fwd = transfer_entropy(source, target)  # source -> target
    te_bwd = transfer_entropy(target, source)  # target -> source
    assert te_fwd > 0.8
    assert te_bwd < 0.1
    assert te_fwd > te_bwd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv --directory . run pytest tests/test_infotheory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.infotheory'`

- [ ] **Step 3: Write minimal implementation**

`src/qot_course/infotheory/__init__.py`:
```python
```

`src/qot_course/infotheory/classical.py`:
```python
"""Classical information theory: entropy, KL, mutual & transfer information."""

from __future__ import annotations

import numpy as np


def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy H(p) = -sum p log p (in bits by default)."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Kullback-Leibler divergence D(p || q) = sum p log(p/q)."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    mask = p > 0
    if np.any(q[mask] == 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])) / np.log(base))


def mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """Mutual information I(X;Y) from a 2-D joint distribution array."""
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    return float(
        shannon_entropy(px, base) + shannon_entropy(py, base) - shannon_entropy(joint, base)
    )


def conditional_mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """Conditional mutual information I(X;Y|Z) from a 3-D joint array (axes X, Y, Z)."""
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    h_xz = shannon_entropy(joint.sum(axis=1), base)  # marginalise Y
    h_yz = shannon_entropy(joint.sum(axis=0), base)  # marginalise X
    h_z = shannon_entropy(joint.sum(axis=(0, 1)), base)
    h_xyz = shannon_entropy(joint, base)
    return float(h_xz + h_yz - h_z - h_xyz)


def _joint_counts_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Empirical joint distribution of three integer-symbol sequences."""
    n_sym = int(max(x.max(), y.max(), z.max())) + 1
    counts = np.zeros((n_sym, n_sym, n_sym), dtype=float)
    np.add.at(counts, (x, y, z), 1.0)
    return counts


def transfer_entropy(source: np.ndarray, target: np.ndarray, base: float = 2.0) -> float:
    """Transfer entropy TE_{source -> target} = I(target_{t+1}; source_t | target_t).

    For integer-symbol sequences with lag 1 (Schreiber, 2000).
    """
    source = np.asarray(source).astype(int)
    target = np.asarray(target).astype(int)
    target_next = target[1:]
    source_past = source[:-1]
    target_past = target[:-1]
    joint = _joint_counts_3d(target_next, source_past, target_past)  # axes (X=t_next, Y=s_past, Z=t_past)
    return conditional_mutual_information(joint, base)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv --directory . run pytest tests/test_infotheory.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/infotheory/__init__.py src/qot_course/infotheory/classical.py tests/test_infotheory.py
git commit -m "feat(infotheory): add entropy, KL, mutual & transfer information"
```

---

### Task 2: Didactic notebook `s05_information.ipynb`

**Files:**
- Create: `notebooks/s05_information.ipynb`

- [ ] **Step 1: Build the notebook (every figure followed by a "read the figure" cell)**

Section plan:
1. **Title block** + Purpose (this is the spine: relative entropy / KL reappears in Sinkhorn, in quantum Sinkhorn, and in the capstone).
2. **§0 Objectives.**
3. **Setup** (code): seeds, imports, `viz.use_course_style()`.
4. **§1 Surprise & entropy** (md): entropy = average surprise; fair coin = 1 bit.
5. (code) entropy of a biased coin vs bias `p`; plot the concave curve (peak 1 bit at p=0.5).
6. (md) read the figure.
7. **§2 Comparing distributions: KL divergence** (md): D(p||q), asymmetry, "extra bits".
8. (code) compute `kl_divergence(p, q)` and `kl_divergence(q, p)` for two example distributions; show they differ.
9. (md) read: KL is not symmetric; it is the parent of everything later (Sinkhorn, quantum relative entropy).
10. **§3 Mutual information** (md): shared information; 0 iff independent.
11. (code) MI of a 2x2 joint as a function of a coupling parameter (from independent to perfectly correlated); plot MI vs coupling.
12. (md) read: MI rises from 0 (independent) to 1 bit (perfectly correlated).
13. **§4 Directed flow: transfer entropy** (md): conditional MI; Schreiber; direction matters.
14. (code) two coupled binary sequences (target copies source at lag 1); bar chart of TE source→target vs target→source.
15. (md) read: TE is strongly directional — it detects who drives whom.
16. **§5 A word of caution** (md): TE estimation is biased at finite samples; PID (Williams–Beer) is *not* uniquely defined — treat redundancy/synergy claims with care. (Honesty, per the course's standard.)
17. **§6 Dictionary update** (md): mutual information ↔ quantum mutual information.
18. **§7 Exercises.**
19. **Conclusions & references** (Cover & Thomas; Schreiber 2000; MacKay).

- [ ] **Step 2: Execute end-to-end**

Run: `uv --directory . run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/s05_information.ipynb`
Expected: exit 0.

- [ ] **Step 3: Clear outputs**

Run: `uv --directory . run jupyter nbconvert --clear-output --inplace notebooks/s05_information.ipynb`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add notebooks/s05_information.ipynb
git commit -m "feat(s5): add didactic classical-information-theory notebook"
```

---

### Task 3: Summary PDF

**Files:**
- Create: `summaries/build_s05_summary.py`

- [ ] **Step 1: Write the builder**

```python
# summaries/build_s05_summary.py
"""Build the Session 5 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Classical information theory --- the spine}

\textbf{Entropy} $H(p) = -\sum_x p(x)\log p(x)$ is the average surprise (1 bit for a fair coin).

\textbf{KL divergence} $D(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$ measures the extra
bits from using $q$ instead of $p$. It is asymmetric and is the parent of the entropic
regularisation behind Sinkhorn (S10) and its quantum cousin (S14).

\textbf{Mutual information} $I(X;Y) = D(p_{XY}\|p_X p_Y)$ is the shared information; zero iff
independent.

\textbf{Transfer entropy} $\mathrm{TE}_{Y\to X} = I(X_{t+1}; Y_t \mid X_t)$ is directed: it
detects who drives whom. \emph{Caveats:} TE estimates are biased at finite samples, and
partial information decomposition (PID) is not uniquely defined --- treat with care.
"""


def main() -> Path:
    """Build the S5 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 5 --- Classical Information Theory",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s05_summary",
    )


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 2: Run it**

Run: `uv --directory . run python summaries/build_s05_summary.py`
Expected: prints a path ending `summaries/s05_summary.pdf`; the file exists.

- [ ] **Step 3: Commit**

```bash
git add summaries/build_s05_summary.py
git commit -m "feat(s5): add session 5 summary PDF generator"
```

---

### Task 4: Dictionary row

**Files:**
- Modify: `docs/dictionary.tex`

- [ ] **Step 1: Add after the entanglement row**, before the `% one row added per session` comment:

```latex
mutual information $I(X;Y)$ & quantum mutual information $I(A{:}B)$ \\
```

- [ ] **Step 2: Verify it compiles**

Run: `command -v latexmk && (cd docs && latexmk -pdf -interaction=nonstopmode dictionary.tex) || echo "skip"`
Expected: `dictionary.pdf` produced.

- [ ] **Step 3: Commit**

```bash
git add docs/dictionary.tex
git commit -m "docs(s5): add mutual-information row to the dictionary"
```

---

## Self-Review

**1. Spec coverage (S5 row).** Shannon entropy, KL, mutual information, conditional MI, transfer entropy (Task 1 + notebook §1–4); circumspect note on PID and TE estimation (notebook §5); the "KL is the spine" framing pointing forward to Sinkhorn/quantum Sinkhorn (notebook §2, summary). Information geometry is deferred to S6. Deliverable contract met (module/notebook/summary/dictionary; inline figures, no new viz).

**2. Placeholder scan.** Complete runnable code in module/summary; notebook is a concrete section plan with exact helper calls; full markdown at execution. No TBD/TODO.

**3. Type/name consistency.** `shannon_entropy(p, base)`, `kl_divergence(p, q, base)`, `mutual_information(joint, base)`, `conditional_mutual_information(joint, base)`, `transfer_entropy(source, target, base)` defined in Task 1 and used consistently in Task 2. `transfer_entropy` reduces to `conditional_mutual_information` of the (X=target_next, Y=source_past, Z=target_past) joint — axis order matches the CMI definition I(X;Y|Z).

---

*Next: **S6 — Information geometry** (Fisher–Rao metric; the two geometries of the simplex; the Amari bridge teased before S10).*
