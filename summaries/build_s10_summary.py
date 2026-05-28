"""Build the Session 10 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Duality \& Sinkhorn --- where Wasserstein meets KL}

\textbf{Kantorovich duality.} The discrete OT LP
$\min_{P \in T(a, b)} \langle C, P\rangle$ admits the dual
$\max_{\varphi, \psi} \langle a, \varphi\rangle + \langle b, \psi\rangle$ subject to
$\varphi_i + \psi_j \le C_{ij}$. For $c = |x - y|$, the dual is the
\textbf{Kantorovich--Rubinstein} formula
$W_1(\mu, \nu) = \sup_{\mathrm{Lip}(f) \le 1} \int f\,\mathrm{d}(\mu - \nu)$.

\textbf{Entropic regularization} (Cuturi, 2013). Add an entropy bonus:
$$
P^\star_\varepsilon = \arg\min_{P \in T(a, b)}\ \langle C, P\rangle - \varepsilon\, H(P),
\qquad H(P) = -\sum_{ij} P_{ij} \log P_{ij}.
$$
KKT conditions force $P^\star_\varepsilon = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)$ with
the \textbf{Gibbs kernel} $K_{ij} = e^{-C_{ij}/\varepsilon}$ and scaling vectors $u, v$
fixed by the marginal constraints.

\textbf{Sinkhorn algorithm} (Sinkhorn, 1964) --- five lines, $\mathcal{O}(nm)$ per
iteration, geometrically convergent:
$$
v \leftarrow b / (K^\top u), \qquad u \leftarrow a / (K v).
$$
Tuning $\varepsilon$ trades \emph{sharpness} for \emph{speed}: $\varepsilon \to 0$
recovers the LP; $\varepsilon \to \infty$ shrinks the plan to the independent coupling
$a \otimes b$.

\textbf{Amari's bridge.} A one-line algebra reveals
$\varepsilon\,\mathrm{KL}(P \,\|\, K) = \langle C, P\rangle - \varepsilon H(P) + \mathrm{const}$,
so \textbf{Sinkhorn $=$ KL projection of the Gibbs kernel onto $T(a, b)$}: the M3
transport geometry intersects the M2 information geometry. Sinkhorn iterations are
iterative Bregman projections under the KL divergence (multiplicative row/column
rescaling). This bridge survives the lift to density matrices and gives quantum
Sinkhorn in S14, with Umegaki relative entropy in place of classical KL.
(Cuturi, 2013; Peyr\'e \& Cuturi, 2019, chs.~4--5; Amari, 2016, sec.~7.5.)
"""


def main() -> Path:
    """Build the S10 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 10 --- Duality \& Sinkhorn",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s10_summary",
    )


if __name__ == "__main__":
    print(main())
