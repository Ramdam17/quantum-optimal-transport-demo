"""Build the Session 14 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Quantum Sinkhorn --- the operator Amari bridge}

\textbf{Entropic quantum OT} (Peyr\'e--Chizat--Vialard--Schmitzer, 2019). Add a von
Neumann entropy bonus to the SDP cost:
$$
P^\star_\varepsilon = \arg\min_{\rho_{AB} \in \Pi(\rho_A, \rho_B)}\
   \mathrm{tr}(C\,\rho_{AB}) - \varepsilon\,S(\rho_{AB}),
$$
with $S(\rho_{AB}) = -\mathrm{tr}(\rho_{AB} \log \rho_{AB})$. Strict convexity gives a
unique minimiser; the KKT conditions force the plan into a matrix-exponential form,
which cvxpy solves directly via the \texttt{von\_neumann\_entr} atom.

\textbf{Matrix-exponential Gibbs kernel.} The classical entrywise kernel
$K_{ij} = e^{-C_{ij}/\varepsilon}$ lifts to $K = \exp(-C/\varepsilon)$ (matrix
exponential). For diagonal $C$ the two coincide; in general the matrix exponential
introduces operator mixing that is impossible classically.

\textbf{Amari quantum bridge.} A one-line algebra reveals
$$
\varepsilon\,S_{\mathrm{Umegaki}}(\rho_{AB} \,\|\, K)
= \mathrm{tr}(C\,\rho_{AB}) - \varepsilon\,S(\rho_{AB}),
$$
so the entropic-QOT plan is the \textbf{Umegaki relative-entropy projection} of the
matrix-exponential Gibbs kernel onto the coupling polytope. M2 (Umegaki, S7) and M4
(couplings, S13) share a single Bregman geometry --- the same identity as S10 lifted
verbatim to operators.

\textbf{The $\varepsilon$ trade-off} is operator-faithful: small $\varepsilon$ gives the
sharp SDP solution (S13); large $\varepsilon$ blurs the plan toward the product
coupling $\rho_A \otimes \rho_B$ (the unique max-entropy coupling with given marginals,
by quantum mutual information non-negativity).

\textbf{The course closes here structurally.} Information geometry (KL / Umegaki) and
transport geometry (Wasserstein / quantum OT) are the same geometry, lifted from
probability vectors to density matrices. The Sinkhorn algorithm is the iterative
Bregman projection in either world. (Cuturi, NeurIPS 2013; Peyr\'e et al., 2019;
Trevisan, arXiv:2202.02091, 2022; Amari, 2016, sec.~7.5.)
"""


def main() -> Path:
    """Build the S14 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 14 --- Quantum Sinkhorn",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s14_summary",
    )


if __name__ == "__main__":
    print(main())
