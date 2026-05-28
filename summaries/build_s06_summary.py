"""Build the Session 6 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Information geometry --- two geometries on the simplex}

The simplex of categorical distributions is a curved geometry. Its natural Riemannian
metric is the \textbf{Fisher information metric}
$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\!\left[\partial_i \log p_\theta\,\partial_j \log p_\theta\right]$.
For the Bernoulli family, $I(\theta) = 1/[\theta(1-\theta)]$ --- the metric \emph{blows up}
at the corners, which are infinitely far away.

A famous change of variable $\phi_i = \sqrt{p_i}$ maps the simplex to a piece of the unit
sphere; the \textbf{Fisher--Rao distance} becomes the angle between sqrt-vectors
(Rao 1945; Bhattacharyya 1943):
$$ d_{\text{FR}}(p, q) = 2\arccos\!\left(\sum_i \sqrt{p_i\, q_i}\right) \in [0, \pi]. $$
Geodesics are great-circle arcs (\emph{slerp} on the sphere of $\sqrt{p}$); on the
2-simplex they \emph{bow away from the boundary}, confirming the curvature.

\textbf{Two geometries on one simplex.} The same simplex also carries the
\textbf{Wasserstein} (transport) geometry. Between two well-separated bumps on a line,
the \emph{mixture} (information-side) midpoint $\tfrac{1}{2}(p_0 + p_1)$ is
\textbf{bimodal} --- amplitudes morph bin by bin, ignoring the ground space. The
\emph{Wasserstein} (transport-side) midpoint is a single peak \textbf{slid} to the
midway position --- mass actually moves on the line. Same endpoints, two geodesics,
two answers.

KL/Fisher--Rao is the natural geometry of \emph{statistical} closeness; Wasserstein is
the natural geometry of \emph{spatial} closeness. The whole course --- and the bridge
to QOT --- lives in the dialogue between these two answers. (Amari, 2016;
Nielsen, 2020; Peyr\'e \& Cuturi, 2019, ch.~7; McCann, 1997.)
"""


def main() -> Path:
    """Build the S6 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 6 --- Information Geometry",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s06_summary",
    )


if __name__ == "__main__":
    print(main())
