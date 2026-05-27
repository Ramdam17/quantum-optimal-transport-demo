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
