"""Preview tools for S12 --- expose precisely what classical OT misses on density matrices.

S12 makes the case for quantum optimal transport. Three quick helpers do most of the
work: a **commutator** (the operator measure of how "quantum" two states are together),
a **diagonal extractor** in the computational basis (the naive classical reduction), and
a **same-diagonal predicate** (the diagnostic for "states with identical Z-statistics").
These are deliberately tiny; the actual quantum OT machinery starts in S13.
"""

from __future__ import annotations

import numpy as np


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Operator commutator ``[A, B] = A B - B A``."""
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    return A @ B - B @ A


def commutativity_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of the commutator ``[A, B]``.

    Zero iff ``A`` and ``B`` commute (share an eigenbasis); strictly positive
    otherwise. A *quantitative* measure of "quantum-ness" of a pair of states ---
    in classical probability (commuting diagonal operators) this is always zero.
    """
    return float(np.linalg.norm(commutator(A, B)))


def diagonal_in_computational_basis(rho: np.ndarray) -> np.ndarray:
    """Return the real diagonal of a density matrix in the Z basis.

    This is the **naive** classical reduction of a quantum state to a probability
    vector --- and the source of the diagonal-collapse problem of S12: two density
    matrices with the same diagonal in one basis are *not* the same state. The output
    is contiguous (POT / scipy LP solvers require this).
    """
    return np.ascontiguousarray(np.diag(np.asarray(rho)).real)


def same_diagonal(
    rho: np.ndarray, sigma: np.ndarray, atol: float = 1e-9
) -> bool:
    """Predicate: do ``rho`` and ``sigma`` have identical diagonals in the Z basis?

    When true, classical OT applied to the diagonals returns the *same* number for
    both states --- but the states themselves may differ in coherences (off-diagonals)
    or in spectrum. The motivating example of S12 is ``|+><+|`` vs ``I/2``, both with
    Z-diagonal ``(1/2, 1/2)`` but different states.
    """
    return bool(
        np.allclose(
            diagonal_in_computational_basis(rho),
            diagonal_in_computational_basis(sigma),
            atol=atol,
        )
    )
