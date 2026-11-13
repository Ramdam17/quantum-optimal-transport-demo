"""Check the house rules of the course notebooks without executing them.

``test_notebooks_execute.py`` proves that every cell runs. This module reads the notebooks as
JSON and checks what an execution cannot see: that they are committed without outputs, that
the prose keeps to the voice rules (no banned word, no emoji, no numbering left over from the
old layout), that every figure is followed by its reading, that the fixed frame of a notebook
is there, and that ``llms.txt`` is the output of its generator. The tests are fast and run in
the default suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = ROOT / "notebooks"

# Words the course never uses in prose, because they tell the learner that a step is easy.
BANNED_WORDS = re.compile(r"\b(obviously|simply|trivially|just)\b", re.IGNORECASE)
# Pictographs and dingbats; the mathematical arrows and operators of the prose are outside.
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿⭐⬆]")
# The old layout spoke of "Movement 2" and "notebook 03"; references are written ``02/03``.
STALE_NUMBERING = re.compile(
    r"\bMovement \d|\bnotebook `?\d\d`?(?![\d/_])", re.IGNORECASE
)
# A code cell that draws: it creates a figure or calls a plotting helper of the course.
DRAWS_A_FIGURE = re.compile(
    r"plt\.show\(|plt\.subplots\(|viz\.plot_|\.savefig\(|fig, ax"
)
READING = re.compile(r"\s*\*\*Read (the|both|each|all|this)")

# The closing synthesis of module 04 is a recapitulation, not a lesson: it has its own frame.
FRAME_EXEMPT = {"04_QuantumOptimalTransport/20_frontier_synthesis"}
# The first tour of the library asks nothing of the learner yet.
EXERCISES_EXEMPT = FRAME_EXEMPT | {"00_GettingStarted/02_what_is_optimal_transport"}


def _all_notebooks() -> list[Path]:
    """Return every notebook of the course, the dictionary included, sorted by path."""
    found = NOTEBOOKS_ROOT.glob("**/*.ipynb")
    return sorted(p for p in found if ".ipynb_checkpoints" not in p.parts)


def _lesson_notebooks() -> list[Path]:
    """Return the numbered notebooks of modules 00 to 05, sorted by path."""
    return sorted(NOTEBOOKS_ROOT.glob("0[0-5]_*/[0-9][0-9]_*.ipynb"))


def _cells(path: Path) -> list[tuple[str, str, dict]]:
    """Return ``(cell type, source text, raw cell)`` for each cell of a notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        source = cell["source"]
        text = source if isinstance(source, str) else "".join(source)
        cells.append((cell["cell_type"], text, cell))
    return cells


def _key(path: Path) -> str:
    """Return ``"<module folder>/<file stem>"``, the form used by the exemption sets."""
    return f"{path.parent.name}/{path.stem}"


def _id(path: Path) -> str:
    return str(path.relative_to(ROOT))


ALL = pytest.mark.parametrize("nb_path", _all_notebooks(), ids=_id)
LESSONS = pytest.mark.parametrize("nb_path", _lesson_notebooks(), ids=_id)


def test_exemptions_name_existing_notebooks() -> None:
    """An exemption that names no notebook is a leftover of a renaming."""
    existing = {_key(p) for p in _lesson_notebooks()}
    assert EXERCISES_EXEMPT <= existing


@ALL
def test_notebook_is_committed_without_outputs(nb_path: Path) -> None:
    """A code cell carries neither an output nor an execution count."""
    dirty = [
        index
        for index, (kind, _, cell) in enumerate(_cells(nb_path))
        if kind == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert dirty == [], f"cells with outputs or execution counts: {dirty}"


@ALL
def test_prose_keeps_the_voice_rules(nb_path: Path) -> None:
    """No banned word and no left-over numbering in the prose; no emoji anywhere."""
    problems = []
    for index, (kind, text, _) in enumerate(_cells(nb_path)):
        if kind == "markdown":
            problems += [(index, m.group(0)) for m in BANNED_WORDS.finditer(text)]
            problems += [(index, m.group(0)) for m in STALE_NUMBERING.finditer(text)]
        problems += [(index, m.group(0)) for m in EMOJI.finditer(text)]
    assert problems == [], f"(cell, text) to reword: {problems}"


@LESSONS
def test_every_figure_is_followed_by_its_reading(nb_path: Path) -> None:
    """The cell after a figure is a paragraph that begins with ``**Read the figure.**``.

    Two figures drawn by consecutive cells may share one reading ("Read the two figures").
    """
    cells = _cells(nb_path)
    unread = []
    for index, (kind, text, _) in enumerate(cells):
        if kind != "code" or not DRAWS_A_FIGURE.search(text):
            continue
        after = index + 1
        while (
            after < len(cells)
            and cells[after][0] == "code"
            and DRAWS_A_FIGURE.search(cells[after][1])
        ):
            after += 1
        following = cells[after] if after < len(cells) else ("", "", {})
        if following[0] != "markdown" or not READING.match(following[1]):
            unread.append(index)
    assert unread == [], f"figure cells with no reading after them: {unread}"


@LESSONS
def test_notebook_has_the_fixed_frame(nb_path: Path) -> None:
    """A lesson ends with exercises, a recap of what was built, and its references."""
    prose = "\n".join(text for kind, text, _ in _cells(nb_path) if kind == "markdown")
    expected = []
    if _key(nb_path) not in FRAME_EXEMPT:
        expected += ["## What you built", "## References"]
    if _key(nb_path) not in EXERCISES_EXEMPT:
        expected += ["## Your turn"]
    missing = [heading for heading in expected if heading not in prose]
    assert missing == [], f"missing headings: {missing}"


def test_llms_txt_is_the_output_of_its_generator() -> None:
    """``llms.txt`` is regenerated whenever a notebook title or purpose changes."""
    from scripts.gen_llms_txt import build_llms_txt

    committed = (ROOT / "llms.txt").read_text(encoding="utf-8")
    assert committed == build_llms_txt(), "run: uv run python scripts/gen_llms_txt.py"
