"""Build the Session 2 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Qubits and states --- the essentials}

\textbf{State.} A qubit is a unit vector $|\psi\rangle = \cos(\theta/2)|0\rangle +
e^{i\phi}\sin(\theta/2)|1\rangle$, i.e.\ a point on the \emph{Bloch sphere}.

\textbf{Born rule.} Measuring in the computational basis gives outcome $x$ with
probability $|\langle x|\psi\rangle|^2$. Measurement is intrinsically random; finite
shots fluctuate around these probabilities (\emph{shot noise}).

\textbf{Why it matters.} Real hardware returns slightly-off probabilities (read-out
error): prepared states are never perfectly pure --- the motivation for density
matrices (Session 3).
"""


def main() -> Path:
    """Build the S2 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 2 --- Qubits and States",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s02_summary",
    )


if __name__ == "__main__":
    print(main())
