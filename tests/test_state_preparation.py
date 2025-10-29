"""
Tests for quantum state preparation module.

This module tests the StatePreparation class and encoding functions
for preparing quantum states from classical probability distributions.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.quantum.circuits import QuantumCircuitBuilder
from src.quantum.state_preparation import (StatePreparation,
                                           amplitude_encoding, angle_encoding,
                                           basis_encoding)


class TestStatePreparation:
    """Test suite for StatePreparation class."""

    def test_initialization_default(self):
        """Test default initialization."""
        prep = StatePreparation()

        assert prep.normalization == "l1"
        assert prep.tolerance == 1e-6

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        prep = StatePreparation(normalization="l2", tolerance=1e-8)

        assert prep.normalization == "l2"
        assert prep.tolerance == 1e-8

    def test_initialization_invalid_normalization(self):
        """Test error on invalid normalization."""
        with pytest.raises(ValueError, match="Invalid normalization"):
            StatePreparation(normalization="invalid")

    def test_prepare_state_amplitude(self):
        """Test amplitude encoding."""
        prep = StatePreparation()
        probs = np.array([0.5, 0.5])

        circuit = prep.prepare_state(probs, method="amplitude")

        assert isinstance(circuit, QuantumCircuitBuilder)
        assert circuit.circuit.num_qubits == 1

    def test_prepare_state_invalid_method(self):
        """Test error on invalid method."""
        prep = StatePreparation()
        probs = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="Invalid method"):
            prep.prepare_state(probs, method="invalid")

    def test_prepare_state_invalid_distribution(self):
        """Test error on invalid distribution."""
        prep = StatePreparation()

        # 2D array
        with pytest.raises(ValueError, match="must be 1D"):
            prep.prepare_state(np.array([[0.5, 0.5]]))

        # Negative values
        with pytest.raises(ValueError, match="non-negative"):
            prep.prepare_state(np.array([0.5, -0.5]))

        # All zeros
        with pytest.raises(ValueError, match="all zeros"):
            prep.prepare_state(np.array([0.0, 0.0]))

    def test_amplitude_encoding_uniform(self):
        """Test amplitude encoding with uniform distribution."""
        prep = StatePreparation()
        probs = np.array([0.25, 0.25, 0.25, 0.25])

        circuit = prep.prepare_state(probs, method="amplitude")

        # Get statevector
        sv = Statevector.from_instruction(circuit.circuit)
        measured_probs = np.abs(sv.data) ** 2

        # Check probabilities match
        for i in range(4):
            assert np.isclose(measured_probs[i], 0.25, atol=1e-6)

    def test_amplitude_encoding_non_uniform(self):
        """Test amplitude encoding with non-uniform distribution."""
        prep = StatePreparation()
        probs = np.array([0.6, 0.4])

        circuit = prep.prepare_state(probs, method="amplitude")

        sv = Statevector.from_instruction(circuit.circuit)
        measured_probs = np.abs(sv.data) ** 2

        assert np.isclose(measured_probs[0], 0.6, atol=1e-6)
        assert np.isclose(measured_probs[1], 0.4, atol=1e-6)

    def test_amplitude_encoding_padding(self):
        """Test amplitude encoding with padding to power of 2."""
        prep = StatePreparation()
        probs = np.array([0.5, 0.3, 0.2])  # 3 values -> pad to 4

        circuit = prep.prepare_state(probs, method="amplitude", pad_to_power_of_2=True)

        # Should use 2 qubits (4 states)
        assert circuit.circuit.num_qubits == 2

        sv = Statevector.from_instruction(circuit.circuit)
        measured_probs = np.abs(sv.data) ** 2

        # First 3 probabilities should be normalized
        total = 0.5 + 0.3 + 0.2
        assert np.isclose(measured_probs[0], 0.5 / total, atol=1e-5)
        assert np.isclose(measured_probs[1], 0.3 / total, atol=1e-5)
        assert np.isclose(measured_probs[2], 0.2 / total, atol=1e-5)

    def test_amplitude_encoding_l2_normalization(self):
        """Test amplitude encoding with L2 normalization."""
        prep = StatePreparation(normalization="l2")
        amplitudes = np.array([0.6, 0.8])  # Already normalized: 0.6^2 + 0.8^2 = 1

        circuit = prep.prepare_state(amplitudes, method="amplitude")

        sv = Statevector.from_instruction(circuit.circuit)
        measured_probs = np.abs(sv.data) ** 2

        assert np.isclose(measured_probs[0], 0.36, atol=1e-6)
        assert np.isclose(measured_probs[1], 0.64, atol=1e-6)

    def test_basis_encoding(self):
        """Test basis encoding."""
        prep = StatePreparation()
        data = np.array([0, 1, 2, 3])

        circuit = prep.prepare_state(data, method="basis")

        assert isinstance(circuit, QuantumCircuitBuilder)
        assert circuit.circuit.num_qubits == 2  # log2(4)

    def test_angle_encoding(self):
        """Test angle encoding."""
        prep = StatePreparation()
        data = np.array([0.0, 0.5, 1.0])

        circuit = prep.prepare_state(data, method="angle")

        # One qubit per value
        assert circuit.circuit.num_qubits == 3

    def test_angle_encoding_custom_range(self):
        """Test angle encoding with custom angle range."""
        prep = StatePreparation()
        data = np.array([0.0, 1.0])

        circuit = prep.prepare_state(data, method="angle", angle_range=(0, 2 * np.pi))

        assert circuit.circuit.num_qubits == 2

    def test_validate_state_valid(self):
        """Test validation of valid state."""
        prep = StatePreparation()
        probs = np.array([0.5, 0.5])

        circuit = prep.prepare_state(probs, method="amplitude")
        validation = prep.validate_state(circuit, probs)

        assert validation["valid"]
        assert "probabilities" in validation
        assert "norm" in validation
        assert np.isclose(validation["norm"], 1.0, atol=1e-6)

    def test_validate_state_with_fidelity(self):
        """Test validation with fidelity check."""
        prep = StatePreparation()
        probs = np.array([0.7, 0.3])

        circuit = prep.prepare_state(probs, method="amplitude")
        validation = prep.validate_state(circuit, probs)

        assert validation["valid"]
        assert "fidelity" in validation
        assert validation["fidelity"] > 0.99  # High fidelity

    def test_validate_state_quantum_circuit(self):
        """Test validation with raw QuantumCircuit."""
        prep = StatePreparation()
        probs = np.array([0.5, 0.5])

        builder = prep.prepare_state(probs, method="amplitude")
        validation = prep.validate_state(builder.circuit, probs)

        assert validation["valid"]

    def test_prepare_batch(self):
        """Test preparing batch of states."""
        prep = StatePreparation()
        dists = [np.array([0.6, 0.4]), np.array([0.3, 0.7]), np.array([0.5, 0.5])]

        circuits = prep.prepare_batch(dists, method="amplitude")

        assert len(circuits) == 3
        assert all(isinstance(c, QuantumCircuitBuilder) for c in circuits)

    def test_prepare_batch_empty(self):
        """Test preparing empty batch."""
        prep = StatePreparation()

        circuits = prep.prepare_batch([], method="amplitude")

        assert len(circuits) == 0

    def test_compute_required_qubits_amplitude(self):
        """Test computing required qubits for amplitude encoding."""
        prep = StatePreparation()

        assert prep.compute_required_qubits(2, method="amplitude") == 1
        assert prep.compute_required_qubits(4, method="amplitude") == 2
        assert prep.compute_required_qubits(8, method="amplitude") == 3
        assert prep.compute_required_qubits(100, method="amplitude") == 7

    def test_compute_required_qubits_angle(self):
        """Test computing required qubits for angle encoding."""
        prep = StatePreparation()

        assert prep.compute_required_qubits(5, method="angle") == 5
        assert prep.compute_required_qubits(10, method="angle") == 10

    def test_compute_required_qubits_invalid_method(self):
        """Test error on invalid method."""
        prep = StatePreparation()

        with pytest.raises(ValueError, match="Unknown method"):
            prep.compute_required_qubits(10, method="invalid")

    def test_repr(self):
        """Test string representation."""
        prep = StatePreparation(normalization="l2", tolerance=1e-8)

        repr_str = repr(prep)
        assert "StatePreparation" in repr_str
        assert "l2" in repr_str
        assert "1e-08" in repr_str


class TestAmplitudeEncoding:
    """Test suite for amplitude_encoding function."""

    def test_basic_encoding(self):
        """Test basic amplitude encoding."""
        probs = np.array([0.5, 0.5])

        circuit = amplitude_encoding(probs)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 1

    def test_encoding_with_qubits(self):
        """Test encoding with specified qubits."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])

        circuit = amplitude_encoding(probs, n_qubits=2)

        assert circuit.num_qubits == 2

    def test_verify_probabilities(self):
        """Test that encoded probabilities are correct."""
        probs = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = amplitude_encoding(probs)

        sv = Statevector.from_instruction(circuit)
        measured_probs = np.abs(sv.data) ** 2

        # Normalize expected probabilities
        expected = probs / np.sum(probs)
        for i in range(len(expected)):
            assert np.isclose(measured_probs[i], expected[i], atol=1e-6)


class TestBasisEncoding:
    """Test suite for basis_encoding function."""

    def test_basic_encoding(self):
        """Test basic basis encoding."""
        data = np.array([0, 1, 2, 3])

        circuit = basis_encoding(data)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2

    def test_encoding_with_qubits(self):
        """Test encoding with specified qubits."""
        data = np.array([0, 1])

        circuit = basis_encoding(data, n_qubits=3)

        assert circuit.num_qubits == 3


class TestAngleEncoding:
    """Test suite for angle_encoding function."""

    def test_basic_encoding(self):
        """Test basic angle encoding."""
        data = np.array([0.0, 0.5, 1.0])

        circuit = angle_encoding(data)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 3

    def test_encoding_custom_range(self):
        """Test encoding with custom angle range."""
        data = np.array([0.0, 1.0])

        circuit = angle_encoding(data, angle_range=(0, 2 * np.pi))

        assert circuit.num_qubits == 2

    def test_verify_angles(self):
        """Test that angles are correctly applied."""
        data = np.array([0.0, 1.0])

        circuit = angle_encoding(data, angle_range=(0, np.pi))

        # Circuit should have RY gates
        assert circuit.num_qubits == 2
        # First qubit should be |0⟩ (angle=0)
        # Second qubit should be |1⟩ (angle=π)


class TestIntegration:
    """Integration tests for state preparation."""

    def test_prepare_and_simulate(self):
        """Test preparing state and simulating."""
        from src.quantum.simulators import QuantumSimulator

        prep = StatePreparation()
        probs = np.array([0.6, 0.4])

        circuit = prep.prepare_state(probs, method="amplitude")

        sim = QuantumSimulator(backend="statevector")
        result = sim.run(circuit.circuit)

        # Check probabilities match
        measured = result.probabilities
        assert "0" in measured
        assert "1" in measured
        assert np.isclose(measured["0"], 0.6, atol=0.01)
        assert np.isclose(measured["1"], 0.4, atol=0.01)

    def test_multiple_distributions(self):
        """Test encoding multiple distributions."""
        prep = StatePreparation()
        dists = [
            np.array([0.8, 0.2]),
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
        ]

        circuits = prep.prepare_batch(dists, method="amplitude")

        from src.quantum.simulators import QuantumSimulator

        sim = QuantumSimulator(backend="statevector")

        for i, (circuit, expected) in enumerate(zip(circuits, dists)):
            result = sim.run(circuit.circuit)
            probs = result.probabilities

            # Normalize expected
            expected = expected / np.sum(expected)

            assert np.isclose(probs["0"], expected[0], atol=0.01)
            assert np.isclose(probs["1"], expected[1], atol=0.01)

    def test_large_distribution(self):
        """Test encoding larger distribution."""
        prep = StatePreparation()
        n = 16
        probs = np.random.rand(n)
        probs = probs / np.sum(probs)

        circuit = prep.prepare_state(probs, method="amplitude")

        # Should use 4 qubits
        assert circuit.circuit.num_qubits == 4

        # Validate
        validation = prep.validate_state(circuit, probs)
        assert validation["valid"]

    def test_different_encodings_comparison(self):
        """Test different encoding methods."""
        prep = StatePreparation()
        data = np.array([0.25, 0.25, 0.25, 0.25])

        # Amplitude encoding
        circuit_amp = prep.prepare_state(data, method="amplitude")
        assert circuit_amp.circuit.num_qubits == 2

        # Angle encoding
        circuit_angle = prep.prepare_state(data, method="angle")
        assert circuit_angle.circuit.num_qubits == 4  # One per value
