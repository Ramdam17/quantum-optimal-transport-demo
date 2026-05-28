"""Build the Session 8 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Monge $\to$ Kantorovich --- the discrete OT linear program}

Given a source $\mu = \sum_i a_i \delta_{x_i}$ and target $\nu = \sum_j b_j \delta_{y_j}$
with $a \in \Delta^n$, $b \in \Delta^m$, and a ground cost $c(x, y)$.

\textbf{Monge (1781).} Find a transport map $T : X \to Y$ pushing $\mu$ onto $\nu$:
$$ \min_{T : T_{\#}\mu = \nu} \int c(x, T(x))\,\mathrm{d}\mu(x). $$
A function $T$ assigns each source atom to a \emph{single} destination, so the problem
is \textbf{infeasible} whenever some source mass must \emph{split} across multiple
targets (e.g.\ 2 source atoms, 3 target atoms with distinct masses).

\textbf{Kantorovich (1942).} Relax to a coupling $P \in \mathbb{R}_+^{n \times m}$ and
solve the linear program
$$
\min_{P \ge 0}\ \langle C, P \rangle = \sum_{i, j} P_{ij}\, C_{ij}
\quad \text{s.t.} \quad
P\,\mathbf{1} = a, \;\; P^\top \mathbf{1} = b.
$$
The feasible set --- the \textbf{transportation polytope} $T(a, b)$ --- is never empty
(the independent coupling $a\,b^\top$ lives in it), so Kantorovich is \emph{always}
solvable, including the cases where Monge fails. The 2-source / 3-target canonical
example admits the optimum
$P^\star = \begin{pmatrix} 1/3 & 1/6 & 0 \\ 0 & 1/6 & 1/3 \end{pmatrix}$,
splitting each source between two targets. A basic feasible solution has at most
$n + m - 1$ non-zero entries.

\textbf{Birkhoff--von Neumann (1946).} When $a = b = \mathbf{1}/n$, $T(a, b)$ becomes
the \textbf{Birkhoff polytope} $B_n$ of doubly stochastic matrices, whose extreme points
are exactly the $n!$ permutation matrices; every $M \in B_n$ is a convex combination of
permutations. Discrete OT then collapses to the \textbf{assignment problem}
$\min_{\sigma} \sum_i C_{i, \sigma(i)}$, solvable in $\mathcal{O}(n^3)$ by the Hungarian
algorithm. (Peyr\'e \& Cuturi, 2019, chs.~2--3; Villani, 2003, ch.~1.)
"""


def main() -> Path:
    """Build the S8 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 8 --- Monge $\to$ Kantorovich",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s08_summary",
    )


if __name__ == "__main__":
    print(main())
