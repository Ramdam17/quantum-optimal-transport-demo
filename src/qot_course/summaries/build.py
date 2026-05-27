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
