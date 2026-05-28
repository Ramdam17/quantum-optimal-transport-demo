"""Single-qubit states: construction, Bloch coordinates, and the Born rule."""

from __future__ import annotations

import numpy as np

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)
KET_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
KET_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)

_PAULI = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def qubit_state(theta: float, phi: float = 0.0) -> np.ndarray:
    """Return the pure state cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex
    )


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Return Bloch coordinates (x, y, z) = (<X>, <Y>, <Z>) of a pure qubit state."""
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)
    return np.array(
        [float(np.real(state.conj() @ _PAULI[p] @ state)) for p in ("X", "Y", "Z")]
    )


def born_probabilities(state: np.ndarray) -> dict[str, float]:
    """Born rule in the computational (Z) basis: P(0) = |a0|^2, P(1) = |a1|^2."""
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)
    return {"0": float(abs(state[0]) ** 2), "1": float(abs(state[1]) ** 2)}


def sample_counts(
    state: np.ndarray, shots: int = 1024, seed: int | None = None
) -> dict[str, int]:
    """Simulate ``shots`` computational-basis measurements; return integer counts."""
    probs = born_probabilities(state)
    rng = np.random.default_rng(seed)
    draws = rng.choice(["0", "1"], size=shots, p=[probs["0"], probs["1"]])
    counts = {"0": 0, "1": 0}
    for outcome, n in zip(*np.unique(draws, return_counts=True)):
        counts[str(outcome)] = int(n)
    return counts
