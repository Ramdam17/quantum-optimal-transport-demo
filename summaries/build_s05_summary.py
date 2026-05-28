"""Build the Session 5 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Classical information theory --- the spine}

\textbf{Entropy} $H(p) = -\sum_x p(x)\log p(x)$ is the average surprise (1 bit for a fair coin).

\textbf{KL divergence} $D(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$ measures the extra
bits from using $q$ instead of $p$. It is asymmetric and is the parent of the entropic
regularisation behind Sinkhorn (S10) and its quantum cousin (S14).

\textbf{Mutual information} $I(X;Y) = D(p_{XY}\,\|\,p_X p_Y)$ is the shared information; zero
iff independent.

\textbf{Transfer entropy} $\mathrm{TE}_{Y\to X} = I(X_{t+1}; Y_t \mid X_t)$ is directed: it
detects who drives whom. \emph{Caveats:} TE estimates are biased at finite samples, and
partial information decomposition (PID) is not uniquely defined --- treat with care.
"""


def main() -> Path:
    """Build the S5 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 5 --- Classical Information Theory",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s05_summary",
    )


if __name__ == "__main__":
    print(main())
