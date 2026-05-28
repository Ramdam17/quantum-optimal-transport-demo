"""Quantum information theory: relative entropy, mutual information, Bures.

S7 lifts the classical information toolbox of S5 to density matrices:

- **Umegaki relative entropy** :math:`S(\\rho \\| \\sigma) = \\mathrm{tr}\\,\\rho(\\log\\rho - \\log\\sigma)`
  --- the quantum KL, the parent of every entropic regulariser in QOT.
- **Quantum mutual information** :math:`I(A{:}B) = S(\\rho_A) + S(\\rho_B) - S(\\rho_{AB})`,
  which can be **twice** the classical bound for entangled states.
- **Quantum conditional entropy** :math:`S(A|B) = S(\\rho_{AB}) - S(\\rho_B)`, which can be
  **negative** --- a purely quantum effect (Cerf and Adami, 1997).
- **Bures distance** :math:`d_B(\\rho, \\sigma) = \\sqrt{2(1 - F(\\rho, \\sigma))}` --- the
  quantum lift of the Fisher--Rao distance of S6 (here :math:`F` is the Uhlmann
  fidelity, not its square).

References: M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*
(2010), ch. 11; M. M. Wilde, *Quantum Information Theory* (2017), chs. 11--12;
N. J. Cerf and C. Adami, "Negative entropy and information in quantum mechanics",
Phys. Rev. Lett. 79, 5194 (1997); A. Uhlmann, "The 'transition probability' in the
state space of a *-algebra", Rep. Math. Phys. 9, 273 (1976).
"""

from __future__ import annotations

import numpy as np

from qot_course.quantum.composite import partial_trace
from qot_course.quantum.density import fidelity, von_neumann_entropy


def _symmetrise(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return (rho + rho.conj().T) / 2


def _matrix_log_on_support(eigs: np.ndarray, vecs: np.ndarray, atol: float) -> np.ndarray:
    """Matrix logarithm of a PSD operator, zeroed on its kernel (x log x -> 0)."""
    log_eigs = np.where(eigs > atol, np.log(np.where(eigs > atol, eigs, 1.0)), 0.0)
    return (vecs * log_eigs) @ vecs.conj().T


def quantum_relative_entropy(
    rho: np.ndarray, sigma: np.ndarray, base: float = 2.0, atol: float = 1e-12
) -> float:
    """Umegaki relative entropy ``S(rho || sigma) = tr(rho (log rho - log sigma))``.

    Returns ``+inf`` if the support of ``rho`` is not contained in the support of
    ``sigma`` (the natural extension of the classical KL singularity). Convention:
    ``base=2`` returns bits; pass ``base=np.e`` for nats.
    """
    rho_h = _symmetrise(rho)
    sigma_h = _symmetrise(sigma)
    rho_eigs, rho_vecs = np.linalg.eigh(rho_h)
    sigma_eigs, sigma_vecs = np.linalg.eigh(sigma_h)
    rho_eigs = np.clip(rho_eigs, 0.0, None)
    sigma_eigs = np.clip(sigma_eigs, 0.0, None)

    # Support test: if sigma has zero eigenvalues, rho must vanish on that subspace.
    kernel_mask = sigma_eigs <= atol
    if np.any(kernel_mask):
        kernel = sigma_vecs[:, kernel_mask]
        leak = np.real(np.trace(kernel.conj().T @ rho_h @ kernel))
        if leak > 1e-9:
            return float("inf")

    log_rho = _matrix_log_on_support(rho_eigs, rho_vecs, atol)
    log_sigma = _matrix_log_on_support(sigma_eigs, sigma_vecs, atol)
    value = np.real(np.trace(rho_h @ (log_rho - log_sigma)))
    return float(value / np.log(base))


def quantum_mutual_information(
    rho_ab: np.ndarray, dims: list[int], base: float = 2.0
) -> float:
    """Quantum mutual information ``I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)``.

    For a bipartite product state, ``I(A:B) = 0``. For a Bell state (pure, with each
    marginal maximally mixed), ``I(A:B) = 2 log_2 d_A`` --- which for qubits is
    **two bits**, twice the classical bound (Nielsen and Chuang, eq. 11.78).
    """
    rho_a = partial_trace(rho_ab, keep=[0], dims=dims)
    rho_b = partial_trace(rho_ab, keep=[1], dims=dims)
    return (
        von_neumann_entropy(rho_a, base=base)
        + von_neumann_entropy(rho_b, base=base)
        - von_neumann_entropy(rho_ab, base=base)
    )


def quantum_conditional_entropy(
    rho_ab: np.ndarray, dims: list[int], base: float = 2.0
) -> float:
    """Quantum conditional entropy ``S(A|B) = S(rho_AB) - S(rho_B)``.

    Can be **negative** for entangled states --- impossible classically. For a Bell
    pair, ``S(A|B) = -1`` bit (Cerf and Adami, 1997); this is one of the cleanest
    signatures of entanglement.
    """
    rho_b = partial_trace(rho_ab, keep=[1], dims=dims)
    return von_neumann_entropy(rho_ab, base=base) - von_neumann_entropy(rho_b, base=base)


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Bures distance ``d_B(rho, sigma) = sqrt(2 (1 - F_U(rho, sigma)))``.

    Here ``F_U = tr sqrt(sqrt(rho) sigma sqrt(rho))`` is the **Uhlmann fidelity** (not
    its square --- our package's :func:`fidelity` follows the Nielsen--Chuang
    convention of returning :math:`F_U^2`, which we take a square root of). Bures is
    the quantum analogue of the Fisher--Rao distance of S6: for *diagonal* states it
    reduces to the Bhattacharyya / Hellinger distance on probability vectors.
    """
    f_squared = fidelity(rho, sigma)
    f_uhlmann = float(np.sqrt(max(0.0, min(1.0, f_squared))))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - f_uhlmann))))
