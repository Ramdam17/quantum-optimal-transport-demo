"""Density matrices: construction, purity, entropy, fidelity, trace distance."""

from __future__ import annotations

import numpy as np

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Return the pure-state density matrix rho = |psi><psi|."""
    psi = np.asarray(state, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def maximally_mixed(dim: int = 2) -> np.ndarray:
    """Return the maximally mixed state I / dim."""
    return np.eye(dim, dtype=complex) / dim


def mixed_state(states: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Return the statistical mixture sum_i w_i |psi_i><psi_i| (weights re-normalised)."""
    rho = sum(w * density_matrix(s) for s, w in zip(states, weights))
    return rho / np.trace(rho).real


def _symmetrise(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return (rho + rho.conj().T) / 2


def is_density_matrix(rho: np.ndarray, atol: float = 1e-9) -> bool:
    """Check Hermitian, unit-trace, positive-semidefinite."""
    rho = np.asarray(rho, dtype=complex)
    hermitian = np.allclose(rho, rho.conj().T, atol=atol)
    unit_trace = abs(np.trace(rho) - 1.0) < atol
    psd = bool(np.all(np.linalg.eigvalsh(_symmetrise(rho)) > -atol))
    return bool(hermitian and unit_trace and psd)


def purity(rho: np.ndarray) -> float:
    """Return tr(rho^2) in [1/dim, 1]; equals 1 iff pure."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """Return S(rho) = -tr(rho log rho), in bits by default."""
    vals = np.linalg.eigvalsh(_symmetrise(rho))
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log(vals)) / np.log(base))


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a Hermitian positive-semidefinite matrix (robust to singularity)."""
    vals, vecs = np.linalg.eigh(_symmetrise(matrix))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    sqrt_rho = _sqrtm_psd(rho)
    inner = _sqrtm_psd(sqrt_rho @ sigma @ sqrt_rho)
    return float(np.real(np.trace(inner)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return T(rho, sigma) = 1/2 ||rho - sigma||_1."""
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    vals = np.linalg.eigvalsh(_symmetrise(diff))
    return float(0.5 * np.sum(np.abs(vals)))


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Bloch vector r = (tr(rho X), tr(rho Y), tr(rho Z)) of a qubit density matrix."""
    rho = np.asarray(rho, dtype=complex)
    return np.array([float(np.real(np.trace(rho @ M))) for M in (_X, _Y, _Z)])


def density_from_bloch(r: np.ndarray) -> np.ndarray:
    """Qubit density matrix rho = 1/2 (I + r_x X + r_y Y + r_z Z)."""
    r = np.asarray(r, dtype=float)
    return 0.5 * (np.eye(2, dtype=complex) + r[0] * _X + r[1] * _Y + r[2] * _Z)
