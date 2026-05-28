"""Build the Session 7 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Quantum information theory --- the dictionary completed}

S5's classical toolbox lifts to density matrices. Three quantities matter most.

\textbf{Umegaki relative entropy} (the quantum KL)
$S(\rho \,\|\, \sigma) = \mathrm{tr}\!\left[\rho\,(\log\rho - \log\sigma)\right] \ge 0$,
zero iff $\rho = \sigma$, asymmetric, and $+\infty$ when $\mathrm{supp}(\rho)
\not\subset \mathrm{supp}(\sigma)$. It separates $|+\rangle\langle +|$ from $I/2$
($S = 1$ bit) even though their $Z$-diagonals are identical --- the coherence is
\emph{seen}, where classical KL was blind. This is the parent of entropic
quantum-OT regularisation (S14).

\textbf{Quantum mutual information}
$I(A{:}B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$. For a Bell pair, $I = \mathbf{2}$ bits
--- \emph{twice} the classical maximum --- because $S(\rho_{AB}) = 0$ while each
marginal is maximally mixed.

\textbf{Quantum conditional entropy}
$S(A|B) = S(\rho_{AB}) - S(\rho_B)$, which for a Bell pair is $\mathbf{-1}$ bit.
\emph{Negative conditional entropy} is a clean signature of entanglement
(Cerf \& Adami, 1997). \emph{Caveat:} $S(A|B) \ge 0$ is necessary but not sufficient
for separability (bound-entangled states exist).

The \textbf{Werner sweep} $\rho(p) = (1-p)|\mathrm{Bell}\rangle\langle\mathrm{Bell}|
+ p\,I_4/4$ visualises both: $I(A{:}B)$ decays smoothly from $2$ to $0$, while
$S(A|B)$ rises from $-1$ through $0$ into the classical regime.

\textbf{Bures distance} $d_B(\rho, \sigma) = \sqrt{2(1 - F_U(\rho, \sigma))}$ with
the Uhlmann fidelity $F_U = \mathrm{tr}\sqrt{\sqrt{\rho}\,\sigma\sqrt{\rho}}$ is the
quantum lift of the Fisher--Rao distance of S6; for diagonal states it reduces to the
Bhattacharyya / Hellinger metric. The infinitesimal version is the quantum Fisher
information. M2 (the spine) is complete; from S8 we leave the information side and
pick up the transport side. (Nielsen \& Chuang, 2010, ch.~11; Wilde, 2017, chs.~11--12;
Cerf \& Adami, 1997; Uhlmann, 1976.)
"""


def main() -> Path:
    """Build the S7 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 7 --- Quantum Information Theory",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s07_summary",
    )


if __name__ == "__main__":
    print(main())
