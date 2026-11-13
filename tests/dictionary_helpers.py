"""Helpers that read the course dictionary and the dictionary cells of notebooks.

The complete classical <-> quantum table lives in ``notebooks/dictionary.ipynb``.
Each notebook shows only the rows it earns, in one cell that links back to it.
Everything here reads notebooks as JSON and executes nothing, so the checks
belong to the fast test suite.

When to use
-----------
Call :func:`check_course` from a test; it returns one sentence per violation.
The smaller functions are exposed so that unit tests can exercise them alone.

Example
-------
>>> split_row(r"| KL $D(p\\|q)$ | Umegaki $S(\\rho\\|\\sigma)$ | `02/02` |")[2]
'`02/02`'
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

LINK_TARGET = "dictionary.ipynb"
PLAIN_PATH = "notebooks/dictionary.ipynb"
# Ceiling over the first two cells of a row. It stops the August rows from
# growing back; see the design note on the dictionary for how it may be raised.
MAX_ROW_WORDS = 45

DICTIONARY_FILE = "dictionary.ipynb"
_ID_PATTERN = re.compile(r"\b(\d{2})/(\d{2})\b")
_SEPARATOR_CELL = re.compile(r"^:?-{1,}:?$")
_MODULE_HEADING = re.compile(r"^#{1,6}\s+Module\s+(\d{2})\b")
# A markdown link whose target ends in the dictionary notebook file name.
_LINK = re.compile(r"\]\([^)\s]*" + re.escape(LINK_TARGET) + r"[^)]*\)")
# The one target a notebook of a module folder may use for that link.
EXPECTED_LINK = "../" + LINK_TARGET
_LINK_TARGET_OF = re.compile(r"\]\(([^)\s]*" + re.escape(LINK_TARGET) + r")[^)]*\)")


@dataclass(frozen=True)
class Row:
    """One row of the complete dictionary.

    Attributes
    ----------
    classical : str
        Normalised text of the "Classical" cell.
    quantum : str
        Normalised text of the "Quantum" cell.
    earned_in : tuple of str
        Notebook identifiers such as ``"02/02"``, in the order written.
    """

    classical: str
    quantum: str
    earned_in: tuple[str, ...]


def split_row(line: str) -> list[str]:
    """Split a markdown table line into stripped cells.

    A vertical bar separates cells unless it is escaped (``\\|``) or sits
    inside a ``$...$`` formula, because the course writes ``|0>`` kets and
    ``D(p\\|q)`` inside cells. Escaped bars are kept as written, so a cell is
    compared with its twin in another notebook character for character.

    Parameters
    ----------
    line : str
        One line of a markdown table, with or without outer bars.

    Returns
    -------
    list of str
        The cells, stripped of surrounding whitespace.
    """
    text = line.strip()
    cells: list[str] = []
    current: list[str] = []
    in_math = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            # Keep the escape and the escaped character together.
            current.append(text[i : i + 2])
            i += 2
            continue
        if char == "$":
            in_math = not in_math
        if char == "|" and not in_math:
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current))
    # Outer bars produce an empty first and last piece; they are not cells.
    if text.startswith("|"):
        cells = cells[1:]
    if text.endswith("|") and not text.endswith("\\|") and cells:
        cells = cells[:-1]
    return [c.strip() for c in cells]


def normalise(cell: str) -> str:
    """Collapse every run of whitespace, line breaks included, to one space."""
    return " ".join(cell.split())


def notebook_id(path: Path) -> str:
    """Return the identifier of a course notebook, such as ``"02/01"``.

    Parameters
    ----------
    path : Path
        ``notebooks/NN_Module/MM_title.ipynb``.
    """
    return f"{path.parent.name.split('_')[0]}/{path.name.split('_')[0]}"


def _markdown_cells(nb_path: Path) -> list[str]:
    """Return the source of every cell as one string ("" for code cells).

    The list is index-aligned with the notebook so that callers can report
    cell positions.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            out.append("".join(cell["source"]))
        else:
            out.append("")
    return out


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL.match(c) for c in cells)


def _tables(source: str) -> list[tuple[list[str], list[list[str]]]]:
    """Find the tables of a markdown source as ``(header, rows)`` pairs.

    A table is a line holding a bar, a separator line, then consecutive lines
    that start with a bar. Cells are returned stripped but not normalised.
    """
    lines = source.splitlines()
    found: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i + 1 < len(lines):
        header = split_row(lines[i]) if "|" in lines[i] else []
        if header and _is_separator(split_row(lines[i + 1])):
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            found.append((header, rows))
            i = j
        else:
            i += 1
    return found


def _is_dictionary_header(header: list[str]) -> bool:
    return [normalise(c) for c in header[:3]] == ["Classical", "Quantum", "Earned in"]


def _is_classical_quantum(header: list[str]) -> bool:
    """True if the first two header cells begin with Classical and Quantum."""
    return (
        len(header) >= 2
        and normalise(header[0]).startswith("Classical")
        and normalise(header[1]).startswith("Quantum")
    )


def _parse_dictionary(path: Path) -> list[tuple[Row, str | None]]:
    """Parse the dictionary notebook into rows with the module heading above them."""
    parsed: list[tuple[Row, str | None]] = []
    module: str | None = None
    for source in _markdown_cells(path):
        # Walk heading lines and tables in document order within the cell.
        lines = source.splitlines()
        i = 0
        while i < len(lines):
            heading = _MODULE_HEADING.match(lines[i])
            if heading:
                module = heading.group(1)
            i += 1
        # Headings and tables sit in the same cell or in consecutive cells; a
        # cell with a heading and a table has its heading first, so the module
        # found above is the right one for the tables below.
        for header, rows in _tables(source):
            if not _is_dictionary_header(header):
                continue
            for cells in rows:
                if len(cells) < 3:
                    raise ValueError(f"Row with fewer than three cells: {cells!r}")
                ids = tuple(f"{m}/{n}" for m, n in _ID_PATTERN.findall(cells[2]))
                row = Row(normalise(cells[0]), normalise(cells[1]), ids)
                parsed.append((row, module))
    return parsed


def load_dictionary(path: Path) -> list[Row]:
    """Read every table headed ``Classical | Quantum | Earned in``.

    Parameters
    ----------
    path : Path
        The dictionary notebook.

    Returns
    -------
    list of Row
        The rows in document order, cells normalised.
    """
    return [row for row, _ in _parse_dictionary(path)]


def dictionary_cell(nb_path: Path) -> str | None:
    """Return the markdown of the cell that links to the dictionary, or ``None``.

    The cell is recognised by its link, as the design prescribes.
    """
    for source in _markdown_cells(nb_path):
        if _LINK.search(source):
            return source
    return None


def displayed_rows(nb_path: Path) -> list[tuple[str, str]]:
    """Return the normalised first two cells of the rows a notebook displays.

    Only the dictionary cell is read. A notebook may add a third column of its
    own; it is ignored here on purpose, because only the first two cells must
    match the complete table.
    """
    cell = dictionary_cell(nb_path)
    if cell is None:
        return []
    rows: list[tuple[str, str]] = []
    for header, body in _tables(cell):
        if not _is_classical_quantum(header):
            continue
        for cells in body:
            if len(cells) >= 2:
                rows.append((normalise(cells[0]), normalise(cells[1])))
    return rows


def stray_tables(nb_path: Path) -> list[int]:
    """Return the indexes of cells holding a Classical | Quantum table with no link."""
    stray: list[int] = []
    for index, source in enumerate(_markdown_cells(nb_path)):
        if _LINK.search(source):
            continue
        if any(_is_classical_quantum(header) for header, _ in _tables(source)):
            stray.append(index)
    return stray


def _course_notebooks(notebooks_root: Path) -> dict[str, Path]:
    """Map identifiers to paths for every ``NN_*/MM_*.ipynb`` under the root."""
    found: dict[str, Path] = {}
    for path in sorted(notebooks_root.glob("[0-9][0-9]_*/[0-9][0-9]_*.ipynb")):
        found[notebook_id(path)] = path
    return found


def _word_count(row: Row) -> int:
    return len(row.classical.split()) + len(row.quantum.split())


def check_course(
    notebooks_root: Path,
    preview_only: Mapping[str, str],
    not_dictionary: Mapping[str, str],
    legacy: Collection[str],
) -> list[str]:
    """Check the dictionary notebook and every notebook against each other.

    Parameters
    ----------
    notebooks_root : Path
        The ``notebooks`` folder; it holds ``dictionary.ipynb`` and one folder
        per module.
    preview_only : mapping of str to str
        Notebooks that display rows without earning them (identifier to
        reason). They are exempt from both "Earned in" checks.
    not_dictionary : mapping of str to str
        Notebooks that hold a Classical | Quantum table which is not the
        dictionary (identifier to reason). They are exempt from the stray-table
        check only.
    legacy : collection of str
        Notebooks that still hold an old table. They are exempt from the
        stray-table check and from the obligation to display the rows that
        name them, and from nothing else.

    Returns
    -------
    list of str
        One sentence per violation; empty when the course is consistent.
    """
    problems: list[str] = []
    parsed = _parse_dictionary(notebooks_root / DICTIONARY_FILE)
    notebooks = _course_notebooks(notebooks_root)
    table: dict[tuple[str, str], Row] = {}
    for row, _ in parsed:
        key = (row.classical, row.quantum)
        if key in table:
            problems.append(
                f"Row '{row.classical}' | '{row.quantum}' appears twice in the "
                "dictionary notebook."
            )
        table.setdefault(key, row)
    for name, ids in (
        ("preview_only", preview_only),
        ("not_dictionary", not_dictionary),
        ("legacy", legacy),
    ):
        for nb_id in sorted(ids):
            if nb_id not in notebooks:
                problems.append(f"{name} names {nb_id}, which is not a notebook.")

    previous_first = ""
    for row, module in parsed:
        label = f"Row '{row.classical}'"
        if not row.earned_in:
            problems.append(f"{label} names no notebook in its Earned in cell.")
            continue
        if list(row.earned_in) != sorted(row.earned_in):
            problems.append(f"{label} lists its notebooks out of course order.")
        first = row.earned_in[0]
        if first < previous_first:
            problems.append(
                f"{label} (first earned in {first}) comes after a row first "
                f"earned in {previous_first}; rows follow course order."
            )
        previous_first = max(previous_first, first)
        if module != first.split("/")[0]:
            problems.append(
                f"{label} sits under module {module} but is first earned in {first}."
            )
        for nb_id in row.earned_in:
            if nb_id not in notebooks:
                problems.append(f"{label} names {nb_id}, which is not a notebook.")
        words = _word_count(row)
        if words > MAX_ROW_WORDS:
            problems.append(
                f"{label} has {words} words over its first two cells; "
                f"the ceiling is {MAX_ROW_WORDS}."
            )

    for nb_id, path in notebooks.items():
        is_legacy = nb_id in legacy
        cell = dictionary_cell(path)
        if cell is not None:
            for target in _LINK_TARGET_OF.findall(cell):
                if target != EXPECTED_LINK:
                    problems.append(
                        f"{nb_id}: the dictionary cell links to {target}; it must "
                        f"link to exactly {EXPECTED_LINK}."
                    )
        if cell is not None and PLAIN_PATH not in cell:
            problems.append(
                f"{nb_id}: the dictionary cell links to {LINK_TARGET} but does "
                f"not write the plain path {PLAIN_PATH}."
            )
        if not is_legacy and nb_id not in not_dictionary:
            for index in stray_tables(path):
                problems.append(
                    f"{nb_id}: cell {index} holds a Classical | Quantum table "
                    f"without a link to {LINK_TARGET}."
                )

        shown = displayed_rows(path)
        for classical, quantum in shown:
            row = table.get((classical, quantum))
            if row is None:
                problems.append(
                    f"{nb_id}: displays the row '{classical}' | '{quantum}', "
                    "which is not a row of the dictionary notebook."
                )
            elif nb_id not in row.earned_in and nb_id not in preview_only:
                problems.append(
                    f"{nb_id}: displays the row '{classical}', whose Earned in "
                    "cell does not name this notebook."
                )

        if is_legacy or nb_id in preview_only:
            if nb_id in preview_only and not is_legacy and not shown:
                problems.append(f"{nb_id}: is preview-only but displays no row.")
            continue
        shown_set = set(shown)
        for row, _ in parsed:
            if nb_id in row.earned_in and (row.classical, row.quantum) not in shown_set:
                problems.append(
                    f"{nb_id}: does not display the row '{row.classical}', "
                    "which names it."
                )
    return problems
