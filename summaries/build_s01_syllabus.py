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
