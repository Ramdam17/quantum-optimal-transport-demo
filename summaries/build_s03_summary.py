"""Build the Session 3 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Density matrices --- the central object}

\textbf{Definition.} A density matrix $\rho$ is Hermitian, positive-semidefinite, with
$\mathrm{tr}\,\rho = 1$. Pure states are $\rho = |\psi\rangle\langle\psi|$
($\mathrm{tr}\,\rho^2 = 1$); mixed states have $\mathrm{tr}\,\rho^2 < 1$.

\textbf{Coherence.} The diagonal of $\rho$ gives the measurement probabilities; the
\emph{off-diagonal} entries encode quantum coherence. $|+\rangle\langle+|$ and the
maximally mixed $I/2$ share a diagonal yet differ off-diagonal --- a Z-measurement cannot
tell them apart. (This gap is the seed of quantum optimal transport.)

\textbf{Measures.} von Neumann entropy $S(\rho) = -\mathrm{tr}(\rho\log\rho)$ (0 for pure,
1 bit for $I/2$); Uhlmann fidelity and trace distance compare states. Noisy hardware
reconstructs a \emph{mixed} $\rho$ (purity $< 1$) --- why $\rho$ is indispensable.
"""


def main() -> Path:
    """Build the S3 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 3 --- Density Matrices",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s03_summary",
    )


if __name__ == "__main__":
    print(main())
