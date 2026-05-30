import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qot_course.hardware import tomography as tomo
from qot_course.hardware.runtime import get_noisy_backend
from qot_course.quantum.composite import bell_state
from qot_course.quantum.density import density_matrix, fidelity
from qot_course.infotheory.quantum import quantum_mutual_information


def _run(circuits_by_setting, backend, shots=1 << 15, seed=7):
    """Run each tomography circuit and return {setting: counts}."""
    out = {}
    for setting, qc in circuits_by_setting.items():
        from qiskit import transpile

        tqc = transpile(qc, backend)
        counts = backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
        out[setting] = counts
    return out


def test_single_qubit_reconstructs_plus_noiseless():
    prep = QuantumCircuit(1)
    prep.h(0)  # |+>
    circuits = tomo.single_qubit_tomography_circuits(prep)
    assert set(circuits) == {"X", "Y", "Z"}
    counts = _run(circuits, AerSimulator())
    rho = tomo.density_from_counts(counts, n_qubits=1)
    ideal = density_matrix(np.array([1, 1], dtype=complex) / np.sqrt(2))
    assert fidelity(rho, ideal) > 0.99
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


def test_density_from_counts_returns_physical_state():
    prep = QuantumCircuit(1)
    prep.h(0)
    counts = _run(tomo.single_qubit_tomography_circuits(prep), get_noisy_backend())
    rho = tomo.density_from_counts(counts, n_qubits=1)
    evals = np.linalg.eigvalsh(rho)
    assert (evals > -1e-9).all()                       # PSD after projection
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)
    assert np.allclose(rho, rho.conj().T, atol=1e-9)   # Hermitian


def test_two_qubit_bell_qmi_near_two_noiseless():
    prep = QuantumCircuit(2)
    prep.h(0)
    prep.cx(0, 1)  # Bell
    circuits = tomo.two_qubit_tomography_circuits(prep)
    assert len(circuits) == 9
    counts = _run(circuits, AerSimulator())
    rho = tomo.density_from_counts(counts, n_qubits=2)
    ideal = density_matrix(bell_state())
    assert fidelity(rho, ideal) > 0.98
    qmi = quantum_mutual_information(rho, dims=[2, 2])  # bits
    assert qmi > 1.9  # ~2 bits, noiseless


def test_two_qubit_bell_qmi_degraded_on_noisy_backend():
    prep = QuantumCircuit(2)
    prep.h(0)
    prep.cx(0, 1)
    counts = _run(tomo.two_qubit_tomography_circuits(prep), get_noisy_backend())
    rho = tomo.density_from_counts(counts, n_qubits=2)
    qmi = quantum_mutual_information(rho, dims=[2, 2])
    assert qmi < 2.0  # decoherence eats the correlations
