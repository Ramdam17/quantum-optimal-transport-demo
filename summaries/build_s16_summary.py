"""Build the Session 16 final-synthesis summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Frontier \& synthesis --- closing the loop}

\textbf{Fifteen sessions, four movements.} M1 built the quantum language (qubits,
density matrices, partial traces, channels). M2 built information geometry (entropy,
KL/Umegaki, Fisher--Rao/Bures, quantum MI, negative conditional entropy). M3 built
classical OT (Monge, Kantorovich, $W_p$, Bures--Wasserstein, Sinkhorn, Amari bridge).
M4 lifted everything to operators: the coupling SDP (S13), quantum Sinkhorn (S14), and
a synthetic Kuramoto capstone (S15).

\textbf{The De Palma et al.\ (2023) VQE limitation theorem.} The same quantum-$W_1$
machinery we built also \emph{forbids} a wide class of naive variational-quantum
algorithms: for cost Hamiltonians with a Lipschitz structure under quantum-$W_1$,
no shallow-depth VQA reaches the ground state by more than a constant-multiplicative
margin. The course's principal tool has both a \emph{constructive} use (the SDP, the
entropic plan, the coupling measures) and a \emph{restrictive} use (no-go theorems for
sloppy quantum-advantage demos). \emph{Building a sharp framework includes building it
sharp enough to recognise its own boundary.}

\textbf{The QOT taxonomy.} Five non-equivalent definitions: coupling SDP
(De Palma--Trevisan, 2021), entropic / Sinkhorn (Peyr\'e et al., 2019), dynamic /
Carlen--Maas (2014), channel-based (Stinespring dilation), qubit-$W_1$ (De Palma,
Marvian, Trevisan, Lloyd, 2021). They agree on Gaussian / commuting states; they
disagree elsewhere. \emph{Trevisan's taxonomy} (arXiv:2202.02091, 2022) names them all.

\textbf{Open problems.} (i) Quantum transfer entropy --- no universally accepted
definition. (ii) Hyperscanning QOT --- does the quantum machinery beat PLV on real
EEG data? (iii) Algorithmic complexity at many-body scale --- the SDP / matrix-exp
scale polynomially in $d_A d_B$; efficient large-scale operator Sinkhorn is open.

\textbf{The course's greatest hits} (S5--S14, replayed in a single
\texttt{course\_greatest\_hits()} call): entropy of a fair coin $= 1$ bit; mixture vs
$W_2$ midpoint mass concentration; Bell QMI $= 2$ bits, $S(A|B) = -1$ bit;
$S(|+\rangle\langle+| \,\|\, I/2) = 1$ bit; $W_2$ closed-form $\equiv$ LP; Bures
$\equiv \sqrt{\text{Bures--Wasserstein matrix term}}$ on density matrices; QOT SDP
distinguishes $|+\rangle\langle+|$ from $I/2$; \textbf{Amari quantum bridge identity}
$\varepsilon\,S_{\mathrm{Umegaki}}(P \,\|\, K) = \mathrm{tr}(C\,P) - \varepsilon\,S(P)$.
All hold simultaneously --- the course's architectural consistency is intact.

\textbf{An honest accounting.} The course delivered a framework, a 21-row dictionary,
and a discipline of \emph{verifying-against-known-limits and honoring open problems}.
It did \emph{not} prove that quantum coupling measures beat PLV on real hyperscanning
data, that the coupling SDP is the canonical quantum OT, or that any naive VQE-style
quantum-advantage claim is valid. The next step is experimental methodology, not more
mathematics. (Trevisan, 2022; De Palma et al., 2023; Villani, 2003; Amari, 2016;
Peyr\'e \& Cuturi, 2019.)

\bigskip
\noindent\emph{End of course.}
"""


def main() -> Path:
    """Build the S16 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 16 --- Frontier \& Synthesis",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s16_summary",
    )


if __name__ == "__main__":
    print(main())
