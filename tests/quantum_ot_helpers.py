"""Random inputs and a solver-call counter shared by the quantum transport tests."""

import cvxpy as cp
import numpy as np


def random_ket(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Normalised complex vector of length ``dim``."""
    ket = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return ket / np.linalg.norm(ket)


def random_full_rank_state(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Full-rank density matrix of shape ``(dim, dim)``."""
    ginibre = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    wishart = ginibre @ ginibre.conj().T
    return wishart / np.trace(wishart).real


def random_hermitian(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Hermitian matrix of shape ``(dim, dim)``."""
    half = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return half + half.conj().T


def count_solver_calls(monkeypatch) -> list:
    """Record every call to ``cvxpy.Problem.solve`` and return the list of records.

    The list is filled as the calls happen, one entry per call, holding the solver name
    that was requested.
    """
    calls = []
    original_solve = cp.Problem.solve

    def counting_solve(self, *args, **kwargs):
        calls.append(kwargs.get("solver"))
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", counting_solve)
    return calls


def matrices_that_are_not_states() -> dict[str, np.ndarray]:
    """Two-level matrices that fail one defining property of a density matrix each."""
    return {
        "negative_eigenvalue": np.diag([1.2, -0.2]).astype(complex),
        "trace_one_half": np.eye(2, dtype=complex) / 4,
        "zero_matrix": np.zeros((2, 2), dtype=complex),
        "not_hermitian": np.array([[0.5, 0.3], [0.0, 0.5]], dtype=complex),
        "trace_four": 2 * np.eye(2, dtype=complex),
    }
