"""Build the Session 15 capstone summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Capstone --- coupling measures on a synthetic Kuramoto dyad}

\textbf{The open question.} Can a quantum-information / quantum-OT coupling measure
on a synthetic oscillator dyad with known injected coupling do at least as well as the
classical baselines (PLV, cosine correlation)? \emph{We do not promise a positive
answer.}

\textbf{The setup.} Two stochastic Kuramoto oscillators with natural frequencies
$\omega_1, \omega_2$ and injected coupling $K$:
$\mathrm{d}\theta_i = (\omega_i + K\sin(\theta_j - \theta_i))\,\mathrm{d}t
+ \sigma\,\mathrm{d}W_i$. From each oscillator's phase $\theta(t)$ build the
phase-coherent qubit state $|\psi(t)\rangle = (|0\rangle + e^{i\theta(t)}|1\rangle)/\sqrt 2$
and time-average the joint product to a bipartite density matrix
$\rho_{AB} = \mathbb{E}_t[|\psi_A(t)\rangle\langle\psi_A(t)| \otimes |\psi_B(t)\rangle\langle\psi_B(t)|]$.
For uncoupled oscillators, $\rho_{AB} \to I_4/4$; for phase-locked oscillators, the
off-block coherence $\rho_{AB}[01, 10] = \langle e^{i(\theta_A - \theta_B)}\rangle / 4$
(exactly the PLV expectation, lifted to operator land).

\textbf{Four measures.} \emph{Quantum mutual information}
$I(A{:}B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$ (information-theoretic, S7).
\emph{Bures-coupling} $d_B(\rho_{AB}, \rho_A \otimes \rho_B)$ (transport-theoretic, S11
bridge). \emph{Phase-locking value} $|\langle e^{i(\theta_A - \theta_B)}\rangle|$
(classical baseline). \emph{Cosine correlation}
$|\mathrm{corr}(\cos\theta_A, \cos\theta_B)|$ (Euclidean baseline). All four are zero
iff the oscillators are decoupled and monotonically increase with $K$ in our synthetic
sweep.

\textbf{Honest caveats.} (i) Estimation bias --- QMI is systematically biased low at
finite samples (Treves \& Panzeri, 1995). (ii) The \emph{direct-sum vs tensor-product
trap}: building a joint covariance of $(\cos\theta_A, \sin\theta_A, \cos\theta_B,
\sin\theta_B)$ gives a 4$\times$4 matrix on $\mathcal{H}_A \oplus \mathcal{H}_B$, not
$\mathcal{H}_A \otimes \mathcal{H}_B$. (iii) Phase-only construction discards amplitude
coupling. (iv) The promised "QOT beats PLV" claim has \emph{not} been demonstrated here
and requires careful experimental methodology, not more mathematics. The capstone is
an \emph{invitation} to that research, not a conclusion. (Kuramoto, 1984; Lachaux et
al., 1999; Trevisan, 2022.)
"""


def main() -> Path:
    """Build the S15 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 15 --- Capstone: Coupling on a Kuramoto Dyad",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s15_summary",
    )


if __name__ == "__main__":
    print(main())
