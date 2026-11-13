"""Consistency of the course dictionary notebook with the notebooks that show it.

Reads notebooks as JSON and executes nothing, so it belongs to the fast suite.
The unit tests build small courses in ``tmp_path``; the last test checks the
real course.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.dictionary_helpers import (
    MAX_ROW_WORDS,
    PLAIN_PATH,
    check_course,
    dictionary_cell,
    displayed_rows,
    load_dictionary,
    normalise,
    notebook_id,
    split_row,
    stray_tables,
)
from tests.dictionary_helpers import _tables

ROOT = Path(__file__).resolve().parents[1]

# Notebooks that display rows without earning one: they announce the thread.
PREVIEW_ONLY = {"00/02": "section 6 announces the thread; no row is earned there"}
# Notebooks whose Classical | Quantum table is not the dictionary.
NOT_DICTIONARY = {"04/04": "side-by-side of the linear and the semidefinite program"}
# Notebooks that still hold an old table. One entry leaves with each commit that
# converts a notebook, and the set ends empty.
LEGACY = frozenset(
    {"00/02"}
    | {f"01/{n:02d}" for n in range(11, 20)}
    | {f"02/{n:02d}" for n in range(1, 7)}
    | {"02/08", "02/12", "03/04", "03/07", "03/11", "03/14"}
    | {"04/03", "04/05", "04/08", "04/16", "04/20"}
)

HEADER = "| Classical | Quantum | Earned in |\n|---|---|---|\n"
BIT = "bit ($0$ or $1$)"
QUBIT = r"qubit $a_0\,|0\rangle + a_1\,|1\rangle$"
KL = r"KL divergence $D(p\|q)$"
UMEGAKI = r"Umegaki relative entropy $S(\rho\|\sigma)$"


def write_notebook(path: Path, cells: list[str]) -> None:
    """Write a markdown-only notebook with one cell per string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": f"c{i}",
                "metadata": {},
                "source": text.splitlines(keepends=True),
            }
            for i, text in enumerate(cells)
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def dictionary_text(rows_by_module: dict[str, list[str]]) -> list[str]:
    """Cells of a dictionary notebook: a title, then a heading and table per module."""
    cells = ["# Dictionary\n\nOpening paragraph.\n"]
    for module, rows in rows_by_module.items():
        cells.append(f"## Module {module} — Name\n\n{HEADER}" + "".join(rows))
    return cells


def row(classical: str, quantum: str, earned: str) -> str:
    return f"| {classical} | {quantum} | {earned} |\n"


def local_cell(rows: list[tuple[str, str]], link: str = PLAIN_PATH) -> str:
    """A notebook dictionary cell: bold lead, link, plain path, table."""
    body = "".join(f"| {c} | {q} |\n" for c, q in rows)
    return (
        f"**Dictionary.** See [the dictionary](../dictionary.ipynb) "
        f"(`{link}`).\n\n| Classical | Quantum |\n|---|---|\n{body}"
    )


def make_course(
    tmp_path: Path,
    table: dict[str, list[str]],
    notebooks: dict[str, list[str]],
) -> Path:
    """Create ``tmp_path/notebooks`` with a dictionary and the given notebooks."""
    root = tmp_path / "notebooks"
    write_notebook(root / "dictionary.ipynb", dictionary_text(table))
    for nb_id, cells in notebooks.items():
        module, number = nb_id.split("/")
        write_notebook(root / f"{module}_Module" / f"{number}_title.ipynb", cells)
    return root


def check(root: Path, preview=None, not_dict=None, legacy=()) -> list[str]:
    return check_course(root, preview or {}, not_dict or {}, legacy)


def good_course(tmp_path: Path) -> Path:
    """A consistent two-notebook course with one row in each."""
    return make_course(
        tmp_path,
        {"01": [row(BIT, QUBIT, "`01/01`")], "02": [row(KL, UMEGAKI, "`02/01`")]},
        {
            "01/01": ["# One\n", local_cell([(BIT, QUBIT)])],
            "02/01": ["# Two\n", local_cell([(KL, UMEGAKI)])],
        },
    )


def test_split_row_keeps_bars_inside_math():
    line = r"| bit ($0$ or $1$) | qubit $a_0\,|0\rangle + a_1\,|1\rangle$ |"
    assert split_row(line) == [
        "bit ($0$ or $1$)",
        r"qubit $a_0\,|0\rangle + a_1\,|1\rangle$",
    ]
    assert (
        len(split_row(r"| KL $D(p\|q)$ | Umegaki $S(\rho\|\sigma)$ | `02/02` |")) == 3
    )
    assert split_row("Classical | Quantum") == ["Classical", "Quantum"]


def test_split_row_keeps_empty_cells_and_inner_cells():
    assert split_row("| a |  | `01/01`, `01/02` |") == ["a", "", "`01/01`, `01/02`"]


def test_normalise_and_notebook_id():
    assert normalise("a  b\n c") == "a b c"
    path = Path("notebooks/02_InformationAndGeometry/01_entropy.ipynb")
    assert notebook_id(path) == "02/01"


def test_load_dictionary_reads_rows_with_math_bars(tmp_path):
    root = make_course(
        tmp_path,
        {
            "01": [row(BIT, QUBIT, "`01/01`")],
            "02": [row(KL, UMEGAKI, "`02/02`, `02/12`")],
        },
        {},
    )
    rows = load_dictionary(root / "dictionary.ipynb")
    assert [r.classical for r in rows] == [BIT, KL]
    assert rows[0].quantum == QUBIT
    assert rows[1].earned_in == ("02/02", "02/12")


def test_dictionary_cell_and_stray_tables(tmp_path):
    path = tmp_path / "01_M" / "01_a.ipynb"
    write_notebook(
        path,
        [
            "# T\n",
            local_cell([(BIT, QUBIT)]),
            "| Classical | Quantum |\n|---|---|\n| a | b |\n",
        ],
    )
    assert "dictionary.ipynb" in (dictionary_cell(path) or "")
    assert displayed_rows(path) == [(BIT, QUBIT)]
    assert stray_tables(path) == [2]


def test_notebook_without_cell_has_no_rows(tmp_path):
    path = tmp_path / "01_M" / "01_a.ipynb"
    write_notebook(path, ["# T\n"])
    assert dictionary_cell(path) is None
    assert displayed_rows(path) == []
    assert stray_tables(path) == []


def test_consistent_course_has_no_violation(tmp_path):
    assert check(good_course(tmp_path)) == []


def test_displayed_row_must_match_the_table(tmp_path):
    table = {"01": [row("first line of a long cell", QUBIT, "`01/01`")]}
    # Same row with different spacing inside a cell (a table row cannot span
    # lines, so a re-wrapped source differs only in its whitespace): accepted.
    wrapped = make_course(
        tmp_path / "a",
        table,
        {"01/01": [local_cell([("first line \t of  a long cell", QUBIT)])]},
    )
    assert check(wrapped) == []
    # One word changed: reported, and the table row is then not displayed.
    changed = make_course(
        tmp_path / "b",
        table,
        {"01/01": [local_cell([("first line of a longer cell", QUBIT)])]},
    )
    problems = check(changed)
    assert any("not a row of the dictionary" in p for p in problems)
    assert any("does not display the row" in p for p in problems)


def test_dictionary_cell_names_the_path(tmp_path):
    root = make_course(
        tmp_path,
        {"01": [row(BIT, QUBIT, "`01/01`")]},
        {"01/01": [local_cell([(BIT, QUBIT)], link="somewhere/else.ipynb")]},
    )
    assert any("plain path" in p for p in check(root))


def test_earned_in_must_name_an_existing_notebook(tmp_path):
    root = make_course(tmp_path, {"01": [row(BIT, QUBIT, "`01/07`")]}, {})
    assert any("01/07" in p and "not a notebook" in p for p in check(root))
    empty = make_course(tmp_path / "e", {"01": [row(BIT, QUBIT, "")]}, {})
    assert any("names no notebook" in p for p in check(empty))


def test_notebook_must_show_every_row_that_names_it(tmp_path):
    root = make_course(
        tmp_path,
        {"01": [row(BIT, QUBIT, "`01/01`"), row("c2", "q2", "`01/01`")]},
        {"01/01": [local_cell([(BIT, QUBIT)])]},
    )
    problems = check(root)
    assert any("does not display the row 'c2'" in p for p in problems)
    # A legacy notebook is exempt from this obligation, a preview-only one too.
    assert check(root, legacy={"01/01"}) == []
    assert check(root, preview={"01/01": "reason"}) == []


def test_notebook_shows_no_row_that_does_not_name_it(tmp_path):
    root = make_course(
        tmp_path,
        {"01": [row(BIT, QUBIT, "`01/01`")], "02": [row(KL, UMEGAKI, "`02/01`")]},
        {
            "01/01": [local_cell([(BIT, QUBIT), (KL, UMEGAKI)])],
            "02/01": [local_cell([(KL, UMEGAKI)])],
        },
    )
    problems = check(root)
    assert any("01/01" in p and "does not name this notebook" in p for p in problems)
    # Preview-only notebooks may display rows they do not earn.
    assert check(root, preview={"01/01": "announces the thread"}) == []


def test_stray_table_is_reported_unless_exempt(tmp_path):
    root = good_course(tmp_path)
    stray = "| Classical | Quantum |\n|---|---|\n| a | b |\n"
    nb = root / "02_Module" / "01_title.ipynb"
    data = json.loads(nb.read_text())
    data["cells"].append(
        {"cell_type": "markdown", "id": "z", "metadata": {}, "source": [stray]}
    )
    nb.write_text(json.dumps(data))
    assert any("without a link" in p for p in check(root))
    assert check(root, not_dict={"02/01": "reason"}) == []
    assert check(root, legacy={"02/01"}) == []


def test_rows_are_in_course_order_and_in_their_module(tmp_path):
    out_of_order = make_course(
        tmp_path / "a",
        {"01": [row("c2", "q2", "`01/02`"), row("c1", "q1", "`01/01`")]},
        {},
    )
    assert any("course order" in p for p in check(out_of_order))
    wrong_module = make_course(tmp_path / "b", {"02": [row("c1", "q1", "`01/01`")]}, {})
    assert any("sits under module 02" in p for p in check(wrong_module))
    unsorted_ids = make_course(
        tmp_path / "c", {"01": [row("c1", "q1", "`01/02`, `01/01`")]}, {}
    )
    assert any("out of course order" in p for p in check(unsorted_ids))


def test_row_over_45_words_is_reported(tmp_path):
    fits = " ".join(["w"] * (MAX_ROW_WORDS - 1))
    over = " ".join(["w"] * MAX_ROW_WORDS)
    table_ok = {"01": [row(fits, "q", "`01/01`")]}
    table_bad = {"01": [row(over, "q", "`01/01`")]}
    ok = make_course(tmp_path / "a", table_ok, {"01/01": [local_cell([(fits, "q")])]})
    bad = make_course(tmp_path / "b", table_bad, {"01/01": [local_cell([(over, "q")])]})
    assert check(ok) == []
    assert any("46 words" in p for p in check(bad))


def test_duplicate_rows_are_reported(tmp_path):
    root = make_course(
        tmp_path,
        {"01": [row(BIT, QUBIT, "`01/01`"), row(BIT, QUBIT, "`01/02`")]},
        {},
    )
    assert any("appears twice" in p for p in check(root))


def test_dictionary_link_must_be_the_module_relative_path(tmp_path):
    def course(target: str) -> Path:
        cell = local_cell([(BIT, QUBIT)]).replace("../dictionary.ipynb", target)
        return make_course(
            tmp_path / target.replace("/", "_"),
            {"01": [row(BIT, QUBIT, "`01/01`")]},
            {"01/01": [cell]},
        )

    assert check(course("../dictionary.ipynb")) == []
    for loose in ("../../notebooks/dictionary.ipynb", "other/dictionary.ipynb"):
        assert any("exactly ../dictionary.ipynb" in p for p in check(course(loose)))


def test_exemptions_must_name_existing_notebooks(tmp_path):
    root = good_course(tmp_path)
    problems = check(
        root,
        preview={"09/09": "r"},
        not_dict={"08/08": "r"},
        legacy={"07/07", "01/01"},
    )
    for ghost in ("09/09", "08/08", "07/07"):
        assert any(ghost in p and "not a notebook" in p for p in problems)
    assert not any("01/01" in p and "not a notebook" in p for p in problems)


def test_real_dictionary_tables_are_well_formed():
    # Passes on the empty table and keeps meaning something once it fills.
    path = ROOT / "notebooks" / "dictionary.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    headers = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for header, _ in _tables("".join(cell["source"])):
            headers += 1
            assert [normalise(c) for c in header] == [
                "Classical",
                "Quantum",
                "Earned in",
            ]
    assert headers >= 1
    for r in load_dictionary(path):
        assert r.classical and r.quantum and r.earned_in


def test_course_dictionary_is_consistent():
    assert check_course(ROOT / "notebooks", PREVIEW_ONLY, NOT_DICTIONARY, LEGACY) == []
