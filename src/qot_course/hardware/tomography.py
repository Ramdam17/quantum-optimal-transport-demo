"""Quantum state tomography from Pauli-basis measurement counts.

Reconstruct a density matrix from the counts returned by measuring a state in
each single-qubit Pauli basis. Used to characterise *real-device* states: the
reconstructed ``rho`` is then fed to :func:`fidelity`,
:func:`quantum_mutual_information`, or :func:`bures_distance` to quantify how far
the hardware state sits from the ideal.

Method: linear inversion ``rho = 2**-n * sum_P <P> P`` over all Pauli strings
``P in {I, X, Y, Z}^{⊗ n}``, followed by projection to the nearest physical
state (Hermitian eigenvalue clip + trace renormalisation). The eigen-clip is the
fast approximation; the trace-norm-optimal projection is Smolin, Gambetta &
Smith (2012), doi:10.1103/PhysRevLett.108.070502.

Conventions
-----------
- Subsystem 0 is qubit 0 and is the most significant factor in ``np.kron``
  (so ``rho`` matches ``quantum_mutual_information(rho, dims=[2, 2])``).
- Qiskit counts are little-endian bitstrings: in a length-``n`` key ``s``, the
  bit of qubit ``k`` is ``s[::-1][k]``.

References
----------
M. A. Nielsen & I. L. Chuang, *Quantum Computation and Quantum Information*,
    ch. 8.4.2 (tomography), Cambridge University Press (2010).
D. F. V. James, P. G. Kwiat, W. J. Munro & A. G. White (2001). Measurement of
    qubits. Physical Review A 64, 052312. doi:10.1103/PhysRevA.64.052312
"""

from __future__ import annotations

import itertools

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _measure_basis(qc: QuantumCircuit, qubit: int, basis: str) -> None:
    """Append the rotation that maps ``basis`` onto the computational (Z) basis."""
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    # "Z": nothing


def single_qubit_tomography_circuits(
    state_prep: QuantumCircuit,
) -> dict[str, QuantumCircuit]:
    """Return the three measurement circuits (keys ``"X"``, ``"Y"``, ``"Z"``).

    Each circuit prepends ``state_prep`` (a 1-qubit, no-measurement circuit),
    rotates to the requested Pauli basis, and measures qubit 0 into creg ``c``.
    """
    circuits = {}
    for basis in ("X", "Y", "Z"):
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(state_prep, qubits=[0], inplace=True)
        _measure_basis(qc, 0, basis)
        qc.measure(0, 0)
        circuits[basis] = qc
    return circuits


def two_qubit_tomography_circuits(
    state_prep: QuantumCircuit,
) -> dict[tuple[str, str], QuantumCircuit]:
    """Return the nine measurement circuits, keyed by ``(basis_q0, basis_q1)``."""
    circuits = {}
    for ba, bb in itertools.product(("X", "Y", "Z"), repeat=2):
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(state_prep, qubits=[0, 1], inplace=True)
        _measure_basis(qc, 0, ba)
        _measure_basis(qc, 1, bb)
        qc.measure([0, 1], [0, 1])
        circuits[(ba, bb)] = qc
    return circuits


def _probs(counts: dict[str, float], n_qubits: int) -> dict[tuple[int, ...], float]:
    """Normalise counts into {(bit_q0, bit_q1, ...): probability}."""
    total = sum(counts.values())
    out = {}
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]  # little-endian: index k -> qubit k
        out[tuple(int(bits[k]) for k in range(n_qubits))] = c / total
    return out


def _project_physical(rho: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix to the nearest PSD, unit-trace density matrix."""
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals.real, 0.0, None)
    if vals.sum() == 0:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    vals = vals / vals.sum()
    return (vecs * vals) @ vecs.conj().T


def density_from_counts(
    counts_by_setting: dict, n_qubits: int
) -> np.ndarray:
    """Reconstruct a density matrix from Pauli-basis measurement counts.

    Parameters
    ----------
    counts_by_setting : dict
        For ``n_qubits=1``: ``{"X": counts, "Y": counts, "Z": counts}``.
        For ``n_qubits=2``: ``{(b0, b1): counts}`` over the 9 settings, with
        ``b0, b1 in {"X","Y","Z"}``. Each ``counts`` maps bitstrings to ints.
    n_qubits : int
        1 or 2.

    Returns
    -------
    np.ndarray
        Reconstructed density matrix, shape ``(2**n, 2**n)``, Hermitian, PSD,
        unit trace.
    """
    if n_qubits == 1:
        exp = {}
        for basis in ("X", "Y", "Z"):
            p = _probs(counts_by_setting[basis], 1)
            exp[basis] = p.get((0,), 0.0) - p.get((1,), 0.0)  # <P> = p0 - p1
        rho = 0.5 * (_I + exp["X"] * _X + exp["Y"] * _Y + exp["Z"] * _Z)
        return _project_physical(rho)

    if n_qubits == 2:
        # Collect Pauli correlators c[(i, j)] = <sigma_i ^ sigma_j>.
        corr: dict[tuple[str, str], float] = {("I", "I"): 1.0}
        single_a: dict[str, list[float]] = {"X": [], "Y": [], "Z": []}
        single_b: dict[str, list[float]] = {"X": [], "Y": [], "Z": []}
        for (ba, bb), counts in counts_by_setting.items():
            p = _probs(counts, 2)
            ea = eb = eab = 0.0
            for (qa, qb), pr in p.items():
                sa, sb = (-1) ** qa, (-1) ** qb
                ea += sa * pr
                eb += sb * pr
                eab += sa * sb * pr
            corr[(ba, bb)] = eab
            single_a[ba].append(ea)
            single_b[bb].append(eb)
        for b in ("X", "Y", "Z"):
            corr[(b, "I")] = float(np.mean(single_a[b]))
            corr[("I", b)] = float(np.mean(single_b[b]))
        rho = np.zeros((4, 4), dtype=complex)
        for i in ("I", "X", "Y", "Z"):
            for j in ("I", "X", "Y", "Z"):
                rho += corr[(i, j)] * np.kron(_PAULI[i], _PAULI[j])
        rho /= 4.0
        return _project_physical(rho)

    raise ValueError(f"n_qubits must be 1 or 2, got {n_qubits}")
