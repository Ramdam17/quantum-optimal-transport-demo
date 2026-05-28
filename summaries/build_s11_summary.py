"""Build the Session 11 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Gaussians \& dynamics --- the Bures bridge into M4}

\textbf{Bures--Wasserstein closed form} (Dowson \& Landau, 1982; Olkin \& Pukelsheim,
1982). For two multivariate Gaussians $\mathcal{N}(m_0, \Sigma_0)$ and
$\mathcal{N}(m_1, \Sigma_1)$ on $\mathbb{R}^d$,
$$
W_2^2 \,=\, \|m_0 - m_1\|^2 + \mathrm{tr}(\Sigma_0) + \mathrm{tr}(\Sigma_1)
       - 2\,\mathrm{tr}\!\sqrt{\Sigma_0^{1/2}\,\Sigma_1\,\Sigma_0^{1/2}}.
$$
The 1-D case collapses to $W_2 = \sqrt{(\Delta m)^2 + (\Delta\sigma)^2}$ --- the
Wasserstein distance \emph{is} the Euclidean distance in the $(m, \sigma)$ plane.

\textbf{The bridge.} The matrix term is defined for any pair of Hermitian PSD matrices.
Restricted to \emph{unit-trace} matrices --- i.e.\ \emph{density matrices} $\rho_0, \rho_1$
--- it becomes
$$ 1 + 1 - 2\,\mathrm{tr}\sqrt{\sqrt{\rho_0}\rho_1\sqrt{\rho_0}} = 2(1 - F_U) = d_B^2(\rho_0, \rho_1), $$
the squared Bures distance of S7. So the Bures--Wasserstein matrix part is
\textbf{identically} the quantum Bures distance: replace covariance matrices by density
matrices and the same formula defines a quantum Wasserstein (S13--S14).

\textbf{Affine McCann map.} For zero-mean Gaussians the optimal transport map is
$T(x) = A x$ with $A = \Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}
\Sigma_0^{-1/2}$, the unique SPD matrix satisfying $A\Sigma_0 A = \Sigma_1$. The $W_2$
geodesic stays Gaussian: $m_t = (1{-}t)m_0 + t m_1$, $\Sigma_t = M_t \Sigma_0 M_t^\top$
with $M_t = (1{-}t)I + tA$ --- translation, rotation, and rescaling in one closed form.

\textbf{Benamou--Brenier} (2000): $W_2^2 = \inf_{(\rho_t, v_t)} \int_0^1\!\int |v_t|^2
\rho_t\,\mathrm{d}x\,\mathrm{d}t$ s.t.\ continuity equation. \textbf{Otto (2001)}: this
makes $\mathcal{P}_2$ a Riemannian space; geodesics are McCann interpolations. In S14 the
continuity equation lifts to a Lindblad-type evolution on density matrices and Otto's
metric to a quantum Riemannian metric. M3 ends; M4 starts at S12.
(Bhatia, Jain \& Lim, 2019.)
"""


def main() -> Path:
    """Build the S11 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 11 --- Gaussians \& Dynamics",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s11_summary",
    )


if __name__ == "__main__":
    print(main())
