"""Composite quantum systems: tensor products, partial trace, entanglement, channels."""

from __future__ import annotations

import numpy as np

from qot_course.quantum.density import von_neumann_entropy

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def tensor(*operands: np.ndarray) -> np.ndarray:
    """Kronecker (tensor) product of states or operators, left to right."""
    result = np.asarray(operands[0], dtype=complex)
    for op in operands[1:]:
        result = np.kron(result, np.asarray(op, dtype=complex))
    return result


def bell_state() -> np.ndarray:
    """Return the Bell state (|00> + |11>) / sqrt(2)."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1.0
    return psi / np.sqrt(2)


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    """Trace out every subsystem not in ``keep``.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix on the composite system.
    keep : list[int]
        Indices of the subsystems to keep.
    dims : list[int]
        Dimension of each subsystem.
    """
    keep = sorted(keep)
    n = len(dims)
    rho = np.asarray(rho, dtype=complex).reshape(dims + dims)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for i in range(n):
        if i not in keep:  # contract row with col -> trace this subsystem out
            col[i] = row[i]
    out = [row[i] for i in keep] + [col[i] for i in keep]
    reduced = np.einsum(rho, row + col, out)
    d_keep = int(np.prod([dims[i] for i in keep]))
    return reduced.reshape(d_keep, d_keep)


def entanglement_entropy(rho: np.ndarray, dims: list[int]) -> float:
    """Entanglement entropy = von Neumann entropy of the first subsystem's reduced state."""
    reduced = partial_trace(rho, keep=[0], dims=dims)
    return von_neumann_entropy(reduced)


def apply_channel(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    """Apply a quantum channel rho -> sum_k K_k rho K_k^dagger."""
    rho = np.asarray(rho, dtype=complex)
    return sum(K @ rho @ K.conj().T for K in kraus)


def is_cptp(kraus: list[np.ndarray], atol: float = 1e-9) -> bool:
    """Check the completeness relation sum_k K_k^dagger K_k = I (trace preservation)."""
    dim = kraus[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus)
    return bool(np.allclose(total, np.eye(dim, dtype=complex), atol=atol))


def depolarizing_channel(p: float) -> list[np.ndarray]:
    """Kraus operators of the single-qubit depolarizing channel with parameter ``p``."""
    return [
        np.sqrt(1.0 - 3.0 * p / 4.0) * _I,
        np.sqrt(p / 4.0) * _X,
        np.sqrt(p / 4.0) * _Y,
        np.sqrt(p / 4.0) * _Z,
    ]
