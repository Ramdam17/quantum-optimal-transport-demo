"""
Tests for quantum circuit simulation module.

This module tests the QuantumSimulator and SimulationResult classes
across different backends and execution modes.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.quantum.circuits import create_bell_state, create_ghz_state
from src.quantum.simulators import QuantumSimulator, SimulationResult


class TestSimulationResult:
    """Test suite for SimulationResult class."""

    def test_statevector_result(self):
        """Test result from statevector simulation."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        assert result.backend_name == "statevector"
        assert result.shots is None
        assert result.statevector is not None
        assert len(result.statevector) == 4
        assert result.probabilities is not None

    def test_qasm_result(self):
        """Test result from qasm simulation."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim = QuantumSimulator(backend="qasm", shots=1000)
        result = sim.run(circuit)

        assert result.backend_name == "qasm"
        assert result.shots == 1000
        assert result.counts is not None
        assert result.probabilities is not None
        assert sum(result.counts.values()) == 1000

    def test_probabilities_from_statevector(self):
        """Test probability extraction from statevector."""
        circuit = QuantumCircuit(1)
        circuit.h(0)  # Equal superposition

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        probs = result.probabilities
        assert "0" in probs
        assert "1" in probs
        assert np.isclose(probs["0"], 0.5, atol=1e-6)
        assert np.isclose(probs["1"], 0.5, atol=1e-6)

    def test_probabilities_from_counts(self):
        """Test probability extraction from counts."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        sim = QuantumSimulator(backend="qasm", shots=10000, seed=42)
        result = sim.run(circuit)

        probs = result.probabilities
        assert "0" in probs
        assert "1" in probs
        # Should be close to 0.5 with many shots
        assert np.isclose(probs["0"], 0.5, atol=0.05)
        assert np.isclose(probs["1"], 0.5, atol=0.05)

    def test_get_probability(self):
        """Test getting probability of specific outcome."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # Bell state: |00⟩ and |11⟩ each with 50% probability
        assert np.isclose(result.get_probability("00"), 0.5, atol=1e-6)
        assert np.isclose(result.get_probability("11"), 0.5, atol=1e-6)
        assert np.isclose(result.get_probability("01"), 0.0, atol=1e-6)
        assert np.isclose(result.get_probability("10"), 0.0, atol=1e-6)

    def test_get_probability_missing(self):
        """Test getting probability of outcome not in results."""
        circuit = QuantumCircuit(1)
        circuit.x(0)  # |1⟩ state

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # State is |1⟩, so |0⟩ has zero probability
        assert result.get_probability("0") == 0.0

    def test_expectation_value(self):
        """Test expectation value calculation."""
        circuit = QuantumCircuit(1)
        circuit.h(0)

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # Pauli Z operator
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        expectation = result.get_expectation_value(pauli_z)

        # For |+⟩ state, <Z> = 0
        assert np.isclose(expectation, 0.0, atol=1e-6)

    def test_expectation_value_requires_statevector(self):
        """Test that expectation value requires statevector."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        sim = QuantumSimulator(backend="qasm")
        result = sim.run(circuit)

        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

        with pytest.raises(ValueError, match="requires statevector"):
            result.get_expectation_value(pauli_z)

    def test_repr(self):
        """Test string representation."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.measure([0, 1], [0, 1])

        sim = QuantumSimulator(backend="qasm", shots=100)
        result = sim.run(circuit)

        repr_str = repr(result)
        assert "SimulationResult" in repr_str
        assert "qasm" in repr_str
        assert "shots=100" in repr_str


class TestQuantumSimulator:
    """Test suite for QuantumSimulator class."""

    def test_initialization_statevector(self):
        """Test initialization with statevector backend."""
        sim = QuantumSimulator(backend="statevector")

        assert sim.backend_name == "statevector"
        assert sim.shots == 1024  # Default
        assert sim.backend is not None

    def test_initialization_qasm(self):
        """Test initialization with qasm backend."""
        sim = QuantumSimulator(backend="qasm", shots=2048, seed=42)

        assert sim.backend_name == "qasm"
        assert sim.shots == 2048
        assert sim.seed == 42

    def test_initialization_invalid_backend(self):
        """Test error on invalid backend."""
        with pytest.raises(ValueError, match="Invalid backend"):
            QuantumSimulator(backend="invalid_backend")

    def test_run_simple_circuit(self):
        """Test running a simple circuit."""
        circuit = QuantumCircuit(1)
        circuit.h(0)

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        assert result is not None
        assert result.statevector is not None
        assert len(result.statevector) == 2

    def test_run_with_measurements(self):
        """Test running circuit with measurements."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim = QuantumSimulator(backend="qasm", shots=1000, seed=42)
        result = sim.run(circuit)

        assert result.counts is not None
        assert sum(result.counts.values()) == 1000
        # Bell state should give roughly equal '00' and '11'
        assert "00" in result.counts
        assert "11" in result.counts

    def test_run_override_shots(self):
        """Test overriding shots parameter."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        sim = QuantumSimulator(backend="qasm", shots=1000)
        result = sim.run(circuit, shots=5000)

        assert result.shots == 5000
        assert sum(result.counts.values()) == 5000

    def test_run_override_seed(self):
        """Test overriding seed parameter."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        sim = QuantumSimulator(backend="qasm", shots=1000)

        result1 = sim.run(circuit, seed=42)
        result2 = sim.run(circuit, seed=42)

        # Same seed should give same results
        assert result1.counts == result2.counts

    def test_run_invalid_input(self):
        """Test error on invalid circuit input."""
        sim = QuantumSimulator(backend="statevector")

        with pytest.raises(ValueError, match="must be a QuantumCircuit"):
            sim.run("not a circuit")

    def test_run_bell_state(self):
        """Test running Bell state circuit."""
        circuit = create_bell_state().circuit

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # Bell state: (|00⟩ + |11⟩)/√2
        sv = result.statevector
        assert np.isclose(abs(sv[0]) ** 2, 0.5, atol=1e-6)  # |00⟩
        assert np.isclose(abs(sv[1]) ** 2, 0.0, atol=1e-6)  # |01⟩
        assert np.isclose(abs(sv[2]) ** 2, 0.0, atol=1e-6)  # |10⟩
        assert np.isclose(abs(sv[3]) ** 2, 0.5, atol=1e-6)  # |11⟩

    def test_run_ghz_state(self):
        """Test running GHZ state circuit."""
        circuit = create_ghz_state(3).circuit

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # GHZ state: (|000⟩ + |111⟩)/√2
        sv = result.statevector
        assert np.isclose(abs(sv[0]) ** 2, 0.5, atol=1e-6)  # |000⟩
        assert np.isclose(abs(sv[7]) ** 2, 0.5, atol=1e-6)  # |111⟩
        # All other amplitudes should be ~0
        for i in [1, 2, 3, 4, 5, 6]:
            assert np.isclose(abs(sv[i]) ** 2, 0.0, atol=1e-6)

    def test_run_batch(self):
        """Test running batch of circuits."""
        circuits = [QuantumCircuit(1), QuantumCircuit(2), QuantumCircuit(1)]
        circuits[0].h(0)
        circuits[1].h(0)
        circuits[1].cx(0, 1)
        circuits[2].x(0)

        sim = QuantumSimulator(backend="statevector")
        results = sim.run_batch(circuits)

        assert len(results) == 3
        assert all(isinstance(r, SimulationResult) for r in results)
        assert len(results[0].statevector) == 2
        assert len(results[1].statevector) == 4
        assert len(results[2].statevector) == 2

    def test_run_batch_empty(self):
        """Test running empty batch."""
        sim = QuantumSimulator(backend="statevector")
        results = sim.run_batch([])

        assert results == []

    def test_get_statevector(self):
        """Test getting statevector directly."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        sim = QuantumSimulator(backend="qasm")  # Note: using qasm backend
        sv = sim.get_statevector(circuit)

        assert isinstance(sv, np.ndarray)
        assert len(sv) == 4
        # Bell state
        assert np.isclose(abs(sv[0]) ** 2, 0.5, atol=1e-6)
        assert np.isclose(abs(sv[3]) ** 2, 0.5, atol=1e-6)

    def test_get_statevector_with_measurements(self):
        """Test getting statevector from circuit with measurements."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim = QuantumSimulator(backend="statevector")
        # Should handle measurements automatically
        sv = sim.get_statevector(circuit)

        assert isinstance(sv, np.ndarray)
        assert len(sv) == 4

    def test_get_counts(self):
        """Test getting counts directly."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim = QuantumSimulator(backend="statevector")  # Note: using statevector
        counts = sim.get_counts(circuit, shots=2000)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 2000
        assert "00" in counts
        assert "11" in counts

    def test_get_counts_no_measurements(self):
        """Test error when getting counts without measurements."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        sim = QuantumSimulator(backend="qasm")

        with pytest.raises(ValueError, match="must have measurements"):
            sim.get_counts(circuit)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with fixed seed."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim1 = QuantumSimulator(backend="qasm", shots=1000, seed=42)
        sim2 = QuantumSimulator(backend="qasm", shots=1000, seed=42)

        result1 = sim1.run(circuit)
        result2 = sim2.run(circuit)

        assert result1.counts == result2.counts

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        sim1 = QuantumSimulator(backend="qasm", shots=1000, seed=42)
        sim2 = QuantumSimulator(backend="qasm", shots=1000, seed=123)

        result1 = sim1.run(circuit)
        result2 = sim2.run(circuit)

        # With high probability, counts should differ
        assert result1.counts != result2.counts

    def test_repr(self):
        """Test string representation."""
        sim = QuantumSimulator(backend="qasm", shots=2048, seed=42)

        repr_str = repr(sim)
        assert "QuantumSimulator" in repr_str
        assert "qasm" in repr_str
        assert "shots=2048" in repr_str
        assert "seed=42" in repr_str


class TestIntegration:
    """Integration tests combining circuits and simulators."""

    def test_parameterized_circuit_execution(self):
        """Test executing parameterized circuits."""
        from qiskit.circuit import Parameter

        theta = Parameter("θ")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        # Bind parameter
        bound_circuit = circuit.assign_parameters({theta: np.pi / 2})

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(bound_circuit)

        # RY(π/2)|0⟩ = |+⟩
        sv = result.statevector
        assert np.isclose(abs(sv[0]) ** 2, 0.5, atol=1e-6)
        assert np.isclose(abs(sv[1]) ** 2, 0.5, atol=1e-6)

    def test_multi_qubit_circuit(self):
        """Test execution of larger circuits."""
        n_qubits = 5
        circuit = QuantumCircuit(n_qubits)

        # Create superposition on all qubits
        for i in range(n_qubits):
            circuit.h(i)

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit)

        # Should have uniform probability over all 2^5 = 32 states
        probs = np.abs(result.statevector) ** 2
        expected_prob = 1.0 / (2**n_qubits)
        assert np.allclose(probs, expected_prob, atol=1e-6)

    def test_statevector_vs_qasm_consistency(self):
        """Test consistency between statevector and qasm backends."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        # Statevector simulation
        sim_sv = QuantumSimulator(backend="statevector")
        # Remove measurements for statevector
        circuit_no_meas = circuit.remove_final_measurements(inplace=False)
        result_sv = sim_sv.run(circuit_no_meas)
        probs_sv = result_sv.probabilities

        # Qasm simulation with many shots
        sim_qasm = QuantumSimulator(backend="qasm", shots=10000, seed=42)
        result_qasm = sim_qasm.run(circuit)
        probs_qasm = result_qasm.probabilities

        # Probabilities should be close
        for key in probs_sv:
            if key in probs_qasm:
                assert np.isclose(
                    probs_sv[key], probs_qasm[key], atol=0.02  # Allow 2% deviation
                )
