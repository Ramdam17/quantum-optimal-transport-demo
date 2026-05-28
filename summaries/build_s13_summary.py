"""Build the Session 13 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Coupling QOT --- a semidefinite program}

The Kantorovich LP of S8 lifts directly to the operator level. A \textbf{quantum
coupling} of $\rho_A \in \mathcal{S}(\mathcal{H}_A)$ and
$\rho_B \in \mathcal{S}(\mathcal{H}_B)$ is a bipartite density matrix
$\rho_{AB}$ on $\mathcal{H}_A \otimes \mathcal{H}_B$ with partial-trace marginals
$\mathrm{tr}_B(\rho_{AB}) = \rho_A$ and $\mathrm{tr}_A(\rho_{AB}) = \rho_B$.
The \textbf{quantum optimal-transport SDP} (De Palma--Trevisan, 2021; Cole, Lostaglio,
Verma, Wilde, 2023) is
$$
\mathrm{QOT}(\rho_A, \rho_B) \,=\, \min_{\rho_{AB} \succeq 0}\
   \mathrm{tr}(C\,\rho_{AB})
   \quad \text{s.t.} \quad
   \mathrm{tr}_B(\rho_{AB}) = \rho_A,\ \mathrm{tr}_A(\rho_{AB}) = \rho_B.
$$
The PSD cone replaces the LP orthant; the rest of the structure transfers verbatim.
\textbf{cvxpy} solves it in five lines.

\textbf{Two canonical cost operators.}
The \textbf{SWAP cost} $C = I - \mathrm{SWAP}$ (Cole et al., 2023) returns
$1 - |\langle\psi|\phi\rangle|^2$ on pure states. The \textbf{quadratic position
cost} $C = (X_A - X_B)^2$ for diagonal $X = \mathrm{diag}(0, 1, \dots, d-1)$ reduces
to the classical $W_2^2$ cost on diagonal marginals --- the diagonal-collapse
principle of S12, made numerical.

\textbf{Three validations.}
(i) \emph{Identity}: $\mathrm{QOT}(\rho, \rho) = 0$ to solver precision.
(ii) \emph{Diagonal collapse}: for commuting diagonal $\rho_A, \rho_B$ the SDP returns
the classical $W_2^2$ on their diagonals --- Trevisan's consistency principle.
(iii) \emph{The $|+\rangle\langle+|$ vs $I/2$ punchline}: classical OT on the
(identical) Z-diagonals returns $0$; the SDP returns a \emph{strictly positive} value
(0.5 with the quadratic cost), exposing exactly what classical OT was structurally
blind to.

\textbf{Pure-state validation.}
For random qubit pairs the SDP returns $1 - |\langle\psi|\phi\rangle|^2$ to within
SDP-solver precision, confirming the Cole et al. closed-form identity.

\textbf{The optimal coupling} $\rho^\star_{AB}$ is itself a bipartite density matrix with
non-trivial off-diagonal structure --- the operator analogue of the mass-flow plot of
S8. M4 is now computational, not just conceptual. (Trevisan, arXiv:2202.02091, 2022;
Boyd \& Vandenberghe, 2004, ch.~11.)
"""


def main() -> Path:
    """Build the S13 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 13 --- Coupling QOT = a Semidefinite Program",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s13_summary",
    )


if __name__ == "__main__":
    print(main())
