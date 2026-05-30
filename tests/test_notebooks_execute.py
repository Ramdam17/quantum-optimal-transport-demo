"""Execute every 00–05 course notebook end-to-end, offline, to guard against silent breakage.

Marked ``notebooks`` (slow): the fast inner loop deselects it with ``-m 'not notebooks'``;
checkpoints run the full suite. Every cell runs on ``AerSimulator`` / ``FakeManilaV2`` (offline) —
no network and no IBM credentials required. The notebook object is executed in memory, so the
committed files stay output-free.

Scope grows by plan: 00–01 (Plan C), 02 (Plan D), 03 (Plan E), 04 (Plan F), 05 (Plan I); later plans widen ``NOTEBOOK_GLOB``.
See ``docs/superpowers/plans/2026-05-28-plan-c-foundations.md`` (decision C-3).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_GLOB = "notebooks/0[012345]_*/[0-9][0-9]_*.ipynb"
CELL_TIMEOUT = 180  # seconds per cell — generous for the qiskit / tomography cells


def _course_notebooks() -> list[Path]:
    """Return the module-00..05 notebooks, sorted by path."""
    return sorted(ROOT.glob(NOTEBOOK_GLOB))


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "nb_path", _course_notebooks(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_notebook_executes_end_to_end(nb_path: Path) -> None:
    """Run every cell of ``nb_path`` on a fresh kernel; fail on any cell error."""
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
