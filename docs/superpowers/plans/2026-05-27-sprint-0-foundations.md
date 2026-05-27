# Sprint 0 — Project Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cross-cutting infrastructure the whole QOT course depends on — a working, tested Python package with logging, config, a graceful-fallback quantum backend, a LaTeX summary pipeline, and living dictionary/glossary skeletons.

**Architecture:** A `src/`-layout package `qot_course` managed by `uv`, with small single-responsibility modules under `utils/`, `hardware/`, and `summaries/`. Everything degrades gracefully offline (the quantum backend defaults to a local simulator; the PDF build is skipped when LaTeX is absent). Tests are pytest, TDD where behaviour exists.

**Tech Stack:** Python 3.12+, `uv`, `numpy`/`scipy`/`matplotlib`, `POT`, `qiskit`/`qiskit-aer`/`qiskit-ibm-runtime`, `cvxpy`, `Jinja2` + `latexmk` (MacTeX), `pytest`/`ruff`/`black`/`mypy`.

**Plan series:** This is **Plan 1 of the course**. Subsequent plans (S1…S16) are written just-in-time and build on this foundation. This plan must leave `uv run pytest` green and a sample summary PDF buildable.

**Verified environment (cross-checked against the owner's working Qiskit tutorials, May 2026):** pinned versions `qiskit==2.4.1`, `qiskit-ibm-runtime==0.47.0`, `qiskit-aer==0.17.2`, plus `pylatexenc` for circuit drawing. Real-hardware idioms to reuse verbatim in later session plans: `QiskitRuntimeService()` → `service.least_busy(operational=True, simulator=False, min_num_qubits=...)`; transpile via `generate_preset_pass_manager(optimization_level=1, backend=backend)`; execute with `SamplerV2`/`EstimatorV2` (options `resilience_level`, `dynamical_decoupling.enable`, `sequence_type="XY4"`); retrieve via `job.job_id()` → `service.job(id)` → `result.data.evs` / `result.data.c.get_counts()`. The free **Open Plan** (`open-instance`) is confirmed working. **Credentials rule (non-negotiable):** never hardcode an API token in code or notebooks — run `QiskitRuntimeService.save_account(...)` once locally (or use an environment variable), then notebooks call `QiskitRuntimeService()` with no arguments.

---

## File structure (created by this plan)

```
quantum-optimal-transport-demo/        # repo root, branch: course
├── pyproject.toml                     # uv project + tooling config
├── .gitignore
├── README.md
├── config/
│   └── base.yaml                      # global defaults (seed, paths)
├── src/qot_course/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py          # get_logger()
│   │   └── config.py                  # load_config(), deep_merge()
│   ├── hardware/
│   │   ├── __init__.py
│   │   └── runtime.py                 # get_backend(), get_noisy_backend()
│   └── summaries/
│       ├── __init__.py
│       ├── build.py                   # build_summary()
│       └── templates/summary.tex.j2   # LaTeX template (Jinja2, LaTeX-safe delimiters)
├── docs/
│   ├── dictionary.tex                 # living classical<->quantum dictionary
│   └── glossary.tex                   # living glossary
└── tests/
    ├── __init__.py
    ├── test_logging_config.py
    ├── test_config.py
    ├── test_runtime.py
    ├── test_build.py
    └── test_smoke.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `config/base.yaml`
- Create: `src/qot_course/__init__.py`, `src/qot_course/utils/__init__.py`, `src/qot_course/hardware/__init__.py`, `src/qot_course/summaries/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "qot-course"
version = "0.1.0"
description = "A hands-on course from classical optimal transport to quantum optimal transport"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "pot>=0.9.3",
    "qiskit>=2.4",
    "qiskit-aer>=0.17.2",
    "qiskit-ibm-runtime>=0.47",
    "cvxpy>=1.4",
    "jinja2>=3.1",
    "pyyaml>=6.0",
    "pylatexenc>=2.10",
    "jupyter>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6", "black>=24.0", "mypy>=1.10"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/qot_course"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.ruff]
line-length = 88
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# LaTeX build artifacts
*.aux
*.log
*.out
*.fls
*.fdb_latexmk
*.synctex.gz

# Data / outputs / secrets
data/
outputs/
logs/
.env
*.key

# OS
.DS_Store
```

- [ ] **Step 3: Create `README.md`**

```markdown
# Quantum Optimal Transport — Hands-On Course

A 16-session hands-on course from classical optimal transport to quantum optimal transport.

- **Teaser (share this):** `docs/qot-course-teaser.md`
- **Design spec:** `docs/superpowers/specs/2026-05-27-qot-course-design.md`

## Setup

```bash
uv sync
uv run pytest
```

All deliverables (code, notebooks, PDF summaries, glossary, dictionary) are in English.
```

- [ ] **Step 4: Create `config/base.yaml`**

```yaml
# Global defaults shared across sessions. Session configs override these.
seed: 42
paths:
  summaries_dir: "summaries"
  docs_dir: "docs"
```

- [ ] **Step 5: Create the five `__init__.py` files**

`src/qot_course/__init__.py`:
```python
"""Quantum Optimal Transport course package."""

__version__ = "0.1.0"
```

`src/qot_course/utils/__init__.py`, `src/qot_course/hardware/__init__.py`, `src/qot_course/summaries/__init__.py`, `tests/__init__.py` — each an empty file:
```python
```

- [ ] **Step 6: Sync the environment and verify collection**

Run: `uv sync && uv run pytest --collect-only`
Expected: environment resolves and installs; pytest collects 0 tests with no import errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md config/base.yaml src/qot_course tests/__init__.py
git commit -m "chore(sprint0): scaffold qot_course package and tooling"
```

---

### Task 2: Logging utility

**Files:**
- Create: `src/qot_course/utils/logging_config.py`
- Test: `tests/test_logging_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_config.py
import logging
from qot_course.utils.logging_config import get_logger


def test_get_logger_returns_named_logger_with_handler():
    logger = get_logger("qot_course.test", level="DEBUG")
    assert logger.name == "qot_course.test"
    assert logger.level == logging.DEBUG
    assert logger.handlers, "logger should have at least one handler"


def test_get_logger_is_idempotent():
    a = get_logger("qot_course.dup")
    b = get_logger("qot_course.dup")
    assert a is b
    assert len(a.handlers) == len(b.handlers)  # no duplicate handlers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.utils.logging_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/utils/logging_config.py
"""Console logging setup for the course. No output is ever silenced."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with a single console handler.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__``).
    level : str
        Logging level, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:  # idempotent: never add duplicate handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/utils/logging_config.py tests/test_logging_config.py
git commit -m "feat(utils): add get_logger with idempotent console handler"
```

---

### Task 3: Config loader

**Files:**
- Create: `src/qot_course/utils/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from qot_course.utils.config import load_config, deep_merge


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\npaths:\n  out: results\n")
    cfg = load_config(p)
    assert cfg["seed"] == 7
    assert cfg["paths"]["out"] == "results"


def test_deep_merge_overrides_nested_keys():
    base = {"seed": 42, "paths": {"a": 1, "b": 2}}
    override = {"paths": {"b": 99}}
    merged = deep_merge(base, override)
    assert merged == {"seed": 42, "paths": {"a": 1, "b": 99}}
    assert base["paths"]["b"] == 2  # input not mutated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.utils.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/utils/config.py
"""YAML config loading and deep-merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` recursively updated by ``override``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/utils/config.py tests/test_config.py
git commit -m "feat(utils): add load_config and deep_merge"
```

---

### Task 4: Quantum backend with graceful fallback

**Files:**
- Create: `src/qot_course/hardware/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py
import qot_course.hardware.runtime as rt
from qiskit_aer import AerSimulator


def test_default_backend_is_local_simulator():
    backend = rt.get_backend()
    assert isinstance(backend, AerSimulator)


def test_prefer_hardware_falls_back_when_service_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(rt, "_make_runtime_service", _raise)
    backend = rt.get_backend(prefer_hardware=True)
    assert isinstance(backend, AerSimulator)  # graceful fallback


def test_noisy_backend_is_runnable():
    backend = rt.get_noisy_backend()
    assert hasattr(backend, "run")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: FAIL with `AttributeError`/`ModuleNotFoundError` (module/functions not defined)

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/hardware/runtime.py
"""Quantum backend selection with graceful fallback to a local simulator.

The course is simulator-first: by default everything runs on AerSimulator,
offline and reproducibly. Real IBM hardware is opt-in and degrades gracefully
to a simulator when no credentials are available.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator

from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)


def _make_runtime_service():
    """Create an IBM Qiskit Runtime service (separated for testability)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService()


def get_backend(prefer_hardware: bool = False):
    """Return a quantum backend.

    With ``prefer_hardware=True``, try the least-busy real QPU; on any failure
    (e.g. missing credentials) fall back to a local ``AerSimulator``.
    """
    if prefer_hardware:
        try:
            service = _make_runtime_service()
            return service.least_busy(operational=True, simulator=False)
        except Exception as exc:  # noqa: BLE001 - we want to fall back on anything
            logger.warning("Hardware unavailable, using simulator: %s", exc)
    return AerSimulator()


def get_noisy_backend(fake: str = "FakeManilaV2"):
    """Return an ``AerSimulator`` carrying a real device's noise model.

    Falls back to a noiseless simulator if the fake backend cannot be loaded.
    """
    try:
        from qiskit_ibm_runtime import fake_provider

        fake_backend = getattr(fake_provider, fake)()
        return AerSimulator.from_backend(fake_backend)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fake backend %s unavailable, using ideal simulator: %s", fake, exc)
        return AerSimulator()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/hardware/runtime.py tests/test_runtime.py
git commit -m "feat(hardware): add backend selection with graceful simulator fallback"
```

---

### Task 5: LaTeX summary pipeline

**Files:**
- Create: `src/qot_course/summaries/templates/summary.tex.j2`
- Create: `src/qot_course/summaries/build.py`
- Test: `tests/test_build.py`

- [ ] **Step 1: Create the LaTeX template (Jinja2 with LaTeX-safe delimiters)**

```latex
% src/qot_course/summaries/templates/summary.tex.j2
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[margin=1in]{geometry}
\title{\VAR{title}}
\author{\VAR{author}}
\date{\VAR{date}}
\begin{document}
\maketitle
\VAR{body}
\end{document}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_build.py
import shutil
import pytest
from qot_course.summaries.build import build_summary

requires_latex = pytest.mark.skipif(
    shutil.which("latexmk") is None, reason="latexmk (MacTeX) not installed"
)


@requires_latex
def test_build_summary_produces_pdf(tmp_path):
    context = {
        "title": "Session 0 --- Roadmap",
        "author": "PPSP lab",
        "date": "2026-05-27",
        "body": "Optimal transport moves mass. We will quantize it.",
    }
    pdf = build_summary(context, out_dir=tmp_path, stem="s00_test")
    assert pdf.exists()
    assert pdf.suffix == ".pdf"
    assert pdf.stat().st_size > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qot_course.summaries.build'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/qot_course/summaries/build.py
"""Render LaTeX summaries from a Jinja2 template and compile with latexmk."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import jinja2

from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# LaTeX-safe delimiters so Jinja does not clash with LaTeX braces.
_ENV = jinja2.Environment(
    block_start_string=r"\BLOCK{",
    block_end_string="}",
    variable_start_string=r"\VAR{",
    variable_end_string="}",
    comment_start_string=r"\#{",
    comment_end_string="}",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
)


def build_summary(
    context: dict[str, Any],
    out_dir: str | Path,
    stem: str,
    template: str = "summary.tex.j2",
) -> Path:
    """Render ``template`` with ``context`` and compile it to ``<stem>.pdf``.

    Returns the path to the generated PDF. Requires ``latexmk`` on PATH.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(_ENV.get_template(template).render(**context), encoding="utf-8")

    logger.info("Compiling %s with latexmk", tex_path)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=out_dir,
        check=True,
    )
    return out_dir / f"{stem}.pdf"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_build.py -v`
Expected: PASS (or SKIPPED if `latexmk` is absent — verified present on the dev machine, so expect PASS)

- [ ] **Step 6: Commit**

```bash
git add src/qot_course/summaries/build.py src/qot_course/summaries/templates/summary.tex.j2 tests/test_build.py
git commit -m "feat(summaries): add Jinja2+latexmk PDF summary builder"
```

---

### Task 6: Living dictionary & glossary skeletons

**Files:**
- Create: `docs/dictionary.tex`, `docs/glossary.tex`

- [ ] **Step 1: Create `docs/dictionary.tex` (compilable, with seed rows)**

```latex
% docs/dictionary.tex --- living classical<->quantum correspondence table.
% Grown one row per session. Compile: latexmk -pdf dictionary.tex
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,booktabs,longtable}
\usepackage[margin=1in]{geometry}
\title{Classical $\leftrightarrow$ Quantum Dictionary}
\date{}
\begin{document}
\maketitle
\begin{longtable}{p{0.45\textwidth} p{0.45\textwidth}}
\toprule
\textbf{Classical} & \textbf{Quantum} \\
\midrule
\endhead
probability vector $p$ & density matrix $\rho$ (diagonal $\Rightarrow$ classical) \\
marginal & partial trace \\
% one row added per session
\bottomrule
\end{longtable}
\end{document}
```

- [ ] **Step 2: Create `docs/glossary.tex` (compilable, with seed entries)**

```latex
% docs/glossary.tex --- living glossary, grown one entry per session.
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[margin=1in]{geometry}
\title{Glossary}
\date{}
\begin{document}
\maketitle
\begin{description}
\item[Optimal transport] The problem of moving mass between two distributions at least cost.
\item[Density matrix] A positive-semidefinite, unit-trace operator describing a quantum state.
% one entry added per session
\end{description}
\end{document}
```

- [ ] **Step 3: Verify both compile (skip if no latexmk)**

Run: `command -v latexmk && (cd docs && latexmk -pdf -interaction=nonstopmode dictionary.tex glossary.tex) || echo "latexmk absent: skip"`
Expected: `dictionary.pdf` and `glossary.pdf` are produced (or the skip message).

- [ ] **Step 4: Commit (source only; build artifacts are git-ignored)**

```bash
git add docs/dictionary.tex docs/glossary.tex
git commit -m "docs: add living dictionary and glossary skeletons"
```

---

### Task 7: Smoke test & green baseline

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_smoke.py
import importlib


def test_all_subpackages_import():
    for module in [
        "qot_course",
        "qot_course.utils.logging_config",
        "qot_course.utils.config",
        "qot_course.hardware.runtime",
        "qot_course.summaries.build",
    ]:
        assert importlib.import_module(module) is not None
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests PASS (the LaTeX test passes on the dev machine; would SKIP elsewhere).

- [ ] **Step 3: Lint & format check**

Run: `uv run ruff check . && uv run black --check src tests`
Expected: no errors (fix any reported issues, then re-run).

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test(sprint0): add import smoke test; green baseline"
```

---

## Self-Review

**1. Spec coverage (Sprint 0 scope).** The spec's §7 repo architecture is scaffolded (package + `utils`/`hardware`/`summaries`); §8 tooling stack is in `pyproject.toml`; §9 hardware track's graceful fallback is `hardware/runtime.py`; §10 dictionary/glossary skeletons created; §11 quality (tests, logging, config, no silenced output) established. Session content (M1–M4) is intentionally deferred to per-session plans — not a gap.

**2. Placeholder scan.** No TBD/TODO; every code step contains complete, runnable code; commands have expected output. The "one row/entry per session" LaTeX comments are intentional extension points, not placeholders.

**3. Type/name consistency.** `get_logger` (Task 2) is reused in Tasks 4 and 5. `_make_runtime_service` is defined in `runtime.py` and monkeypatched by the same name in `test_runtime.py` (Task 4). `build_summary(context, out_dir, stem, template)` signature matches its test call. Package path `qot_course` is consistent across `pyproject.toml` (`packages = ["src/qot_course"]`), imports, and the smoke test.

---

*Next plan: **S1 — Teaser & Roadmap** (roadmap notebook with the visual OT teaser + first `ot.emd2`, an `env_check`, the syllabus PDF built via `build_summary`, initial dictionary rows).*
