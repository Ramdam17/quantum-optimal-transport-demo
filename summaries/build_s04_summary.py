"""Build the Session 4 one-page summary PDF via the summary pipeline."""

from __future__ import annotations

from pathlib import Path

from qot_course.summaries.build import build_summary

_BODY = r"""
\section*{Composite systems and channels}

\textbf{Tensor product.} Two systems combine as $\mathcal{H}_A \otimes \mathcal{H}_B$;
states and operators combine by the Kronecker product.

\textbf{Partial trace} is the quantum marginal: $\rho_A = \mathrm{tr}_B\,\rho_{AB}$ recovers
the state of $A$ alone.

\textbf{Entanglement.} The Bell state $(|00\rangle+|11\rangle)/\sqrt2$ is \emph{pure} yet
its parts are \emph{maximally mixed} ($\rho_A = I/2$, entanglement entropy $= 1$ bit): the
whole is more defined than its parts --- impossible classically.

\textbf{Quantum channels} (the quantum Markov kernels) are completely-positive
trace-preserving maps, $\rho \mapsto \sum_k K_k \rho K_k^\dagger$ with
$\sum_k K_k^\dagger K_k = I$. The depolarizing channel mixes a state toward $I/2$ --- the
noise behind S3's tomography.
"""


def main() -> Path:
    """Build the S4 summary PDF and return its path."""
    return build_summary(
        {
            "title": r"Session 4 --- Composite Systems \& Channels",
            "author": "PPSP lab",
            "date": "2026",
            "body": _BODY,
        },
        out_dir=Path(__file__).parent,
        stem="s04_summary",
    )


if __name__ == "__main__":
    print(main())
