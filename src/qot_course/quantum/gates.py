"""Single- and two-qubit gates and observables as explicit matrices.

The course's operational layer in one place: the named single-qubit gates
(Pauli X/Y/Z, Hadamard, phase S/S-dagger), the parametric rotations
(``phase_gate``, ``rx``/``ry``/``rz``), the two-qubit ``CNOT``, and the helpers
to apply a gate (:func:`apply_gate`), check unitarity (:func:`is_unitary`), and
read an observable's expectation value (:func:`expectation`).

When to use
-----------
Reach for this module whenever a notebook needs to *act on* a state or read an
observable, rather than only describe a state. The constants are the single
source of truth for the Pauli matrices across the course (``quantum.states``
imports them here).

Conventions
-----------
- States are length-2 (single qubit) or length-4 (two qubit) complex vectors;
  gates are ``(2, 2)`` or ``(4, 4)`` complex matrices. ``apply_gate(U, psi)``
  returns ``U @ psi``.
- ``CNOT`` uses qubit 0 as the control and qubit 1 as the target, ordered so
  qubit 0 is the most significant factor in ``np.kron`` — matching the
  convention used throughout :mod:`qot_course.quantum.composite`.

Examples
--------
>>> import numpy as np
>>> from qot_course.quantum.gates import HADAMARD, apply_gate
>>> from qot_course.quantum.states import KET_0, KET_PLUS
>>> bool(np.allclose(apply_gate(HADAMARD, KET_0), KET_PLUS))
True

References
----------
M. A. Nielsen & I. L. Chuang, *Quantum Computation and Quantum Information*,
    ch. 1-2, 4, Cambridge University Press (2010).
"""

from __future__ import annotations

import numpy as np

IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
S_DAG = S.conj().T
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def phase_gate(lmbda: float) -> np.ndarray:
    """Return ``diag(1, e^{i lambda})`` — a relative phase ``lmbda`` on |1>.

    Shape ``(2, 2)``. ``S`` is the special case ``phase_gate(pi / 2)``.
    """
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * lmbda)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    """Rotation by ``theta`` (radians) about the Bloch x-axis: ``exp(-i theta X / 2)``."""
    return np.cos(theta / 2) * IDENTITY - 1j * np.sin(theta / 2) * PAULI_X


def ry(theta: float) -> np.ndarray:
    """Rotation by ``theta`` (radians) about the Bloch y-axis: ``exp(-i theta Y / 2)``."""
    return np.cos(theta / 2) * IDENTITY - 1j * np.sin(theta / 2) * PAULI_Y


def rz(theta: float) -> np.ndarray:
    """Rotation by ``theta`` (radians) about the Bloch z-axis: ``exp(-i theta Z / 2)``."""
    return np.cos(theta / 2) * IDENTITY - 1j * np.sin(theta / 2) * PAULI_Z


def apply_gate(U: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply gate ``U`` to ``state`` and return ``U @ state``.

    No renormalisation is needed because ``U`` is unitary; the output keeps the
    input's norm. Shapes: ``U`` is ``(d, d)``, ``state`` is ``(d,)``.
    """
    return np.asarray(U, dtype=complex) @ np.asarray(state, dtype=complex)


def is_unitary(U: np.ndarray, atol: float = 1e-9) -> bool:
    """Return True iff ``U @ U.conj().T`` equals the identity to ``atol``."""
    U = np.asarray(U, dtype=complex)
    return bool(np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=atol))


def expectation(state: np.ndarray, operator: np.ndarray) -> float:
    """Return the expectation value ``<psi|A|psi>`` (real part; ``A`` Hermitian).

    The state is normalised first. For a Pauli observable this is a Bloch
    coordinate: ``expectation(psi, PAULI_Z)`` equals ``P(0) - P(1)``.
    """
    state = np.asarray(state, dtype=complex)
    state = state / np.linalg.norm(state)
    return float(np.real(state.conj() @ np.asarray(operator, dtype=complex) @ state))
