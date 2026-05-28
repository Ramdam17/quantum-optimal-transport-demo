"""Build the Session 12 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Why quantum optimal transport?}

Classical OT operates on probability vectors. Naively lifting it to density matrices by
"taking the diagonal" fails: the diagonal is \textbf{basis-dependent}, and only the
spectrum is intrinsic. The whole machinery of M4 is the operator-level repair.

\textbf{The canonical example.} $|+\rangle\langle+|$ and $I/2$ both have Z-diagonal
$(1/2, 1/2)$, so classical OT on the diagonals returns $W_p = 0$. But the states are
distinct: $|+\rangle\langle+|$ is pure (entropy $0$, a coherent superposition) and $I/2$
is maximally mixed (entropy $1$ bit, a classical coin flip). The Bures distance and the
Umegaki relative entropy both return strictly positive values. \textbf{The diagonal is
not enough.}

\textbf{Non-commutativity is the deeper source.} For
$\rho_X = |+\rangle\langle+|$ and $\rho_Y = |+i\rangle\langle+i|$, both have Z-diagonal
$(1/2, 1/2)$ \emph{and} $[\rho_X, \rho_Y] \neq 0$. No shared eigenbasis exists, so the
question "which diagonal?" is ill-posed. Quantum OT must be \textbf{intrinsic} to the
operator.

\textbf{Diagonal-collapse consistency principle.} Any candidate quantum OT must reduce
to classical OT whenever $[\rho_0, \rho_1] = 0$ (commuting / simultaneously
diagonalisable). The coupling SDP of S13 satisfies this by construction.

\textbf{Trevisan's taxonomy.} The QOT field has multiple non-equivalent formulations:
\textbf{couplings} (De Palma--Trevisan, 2021 --- the SDP, S13), \textbf{entropic /
quantum Sinkhorn} (Peyr\'e--Cuturi tensor fields, S14), \textbf{dynamic}
(Carlen--Maas, 2014), \textbf{channel-based} (Stinespring dilation), and
\textbf{qubit-$W_1$} (De Palma, Marvian, Trevisan, Lloyd, 2021). This course follows the
coupling SDP. \textbf{The classical $\leftrightarrow$ quantum dictionary is now complete}:
S13 fills the explicit Wasserstein row, S14 the Sinkhorn row, S15--S16 the open frontier.
(Trevisan, arXiv:2202.02091, 2022.)
"""


def main() -> Path:
    """Build the S12 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 12 --- Why Quantum Optimal Transport?",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s12_summary",
    )


if __name__ == "__main__":
    print(main())
