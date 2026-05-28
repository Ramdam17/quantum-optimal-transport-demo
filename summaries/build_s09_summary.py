"""Build the Session 9 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Wasserstein distances --- turning OT cost into a metric}

The S8 Kantorovich cost, taken to a $p$-th root, becomes a genuine metric on
distributions of finite $p$-th moment:
$$
W_p(\mu, \nu) = \Bigl(\inf_{\pi \in \Pi(\mu, \nu)} \int |x - y|^p\, \mathrm{d}\pi(x, y)\Bigr)^{1/p}.
$$
\textbf{Metric axioms} (Villani, 2003, Thm.~7.3): positivity with $W_p(\mu, \nu) = 0
\Leftrightarrow \mu = \nu$, symmetry, and triangle inequality (via the \emph{glueing
lemma} on couplings).

\textbf{1-D closed form} (Brenier; Villani, Thm.~2.18). The optimal map is the quantile
composition $T = F_\nu^{-1} \circ F_\mu$, and
$$
W_p^p(\mu, \nu) = \int_0^1 \bigl|F_\mu^{-1}(u) - F_\nu^{-1}(u)\bigr|^p\, \mathrm{d}u.
$$
For discrete atoms this is a finite sum on the common refinement of the two CDFs,
computable in $\mathcal{O}((n + m)\log(n + m))$ --- the LP collapses to a sort. The gift
\emph{vanishes} in higher dimensions because the quantile function does.

\textbf{$W_1$ as the area between CDFs} (Vallender, 1974):
$W_1(\mu, \nu) = \int |F_\mu(x) - F_\nu(x)|\, \mathrm{d}x$.

\textbf{McCann (1997) displacement geodesic}: $\mu_t = ((1 - t)\,\mathrm{Id} + t\,T)_{\#}\,\mu$,
$t \in [0, 1]$, is the $W_2$-geodesic. Each unit of mass slides linearly from its
$\mu$-position to its $\nu$-position --- the peak \emph{moves}, where the mixture
interpolation (the information-side geodesic) was \emph{bimodal}.

\textbf{Why we need transport.} For a single bump translated by $d$:
$W_2(\mu, \mu_d) = d$ \emph{linearly}, while $D_{\mathrm{KL}}(\mu \| \mu_d) \to \infty$
once supports separate. Wasserstein measures \emph{how far mass must move}; KL is
\emph{blind} to ground geometry. (Peyre \& Cuturi, 2019, ch.~2; Otto, 2001.)
"""


def main() -> Path:
    """Build the S9 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 9 --- Wasserstein Distances",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s09_summary",
    )


if __name__ == "__main__":
    print(main())
