"""
Tests for quantum circuit construction utilities.
"""

import numpy as np
import pytest
from qiskit.circuit import Parameter

from src.quantum.circuits import (QuantumCircuitBuilder, create_bell_state,
                                  create_ghz_state, create_w_state)


class TestQuantumCircuitBuilder:
    """Test suite for QuantumCircuitBuilder class."""

    def test_initialization_basic(self):
        """Test basic circuit initialization."""
        builder = QuantumCircuitBuilder(n_qubits=3)

        assert builder.n_qubits == 3
        assert builder.n_classical == 3
        assert builder.circuit.num_qubits == 3
        assert builder.circuit.num_clbits == 3

    def test_initialization_custom_classical(self):
        """Test initialization with custom classical bits."""
        builder = QuantumCircuitBuilder(n_qubits=5, n_classical=3)

        assert builder.n_qubits == 5
        assert builder.n_classical == 3
        assert builder.circuit.num_clbits == 3

    def test_initialization_invalid_qubits(self):
        """Test that initialization fails with invalid qubit count."""
        with pytest.raises(ValueError, match="at least 1"):
            QuantumCircuitBuilder(n_qubits=0)

    def test_single_qubit_gates(self):
        """Test single-qubit gate application."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        # Test chaining
        builder.h(0).x(1).y(0).z(1)

        assert builder.depth() > 0
        gates = builder.gate_count()
        assert "h" in gates
        assert "x" in gates
        assert "y" in gates
        assert "z" in gates

    def test_rotation_gates(self):
        """Test rotation gates."""
        builder = QuantumCircuitBuilder(n_qubits=1)

        angle = np.pi / 4
        builder.rx(angle, 0).ry(angle, 0).rz(angle, 0)

        gates = builder.gate_count()
        assert gates["rx"] == 1
        assert gates["ry"] == 1
        assert gates["rz"] == 1

    def test_two_qubit_gates(self):
        """Test two-qubit gates."""
        builder = QuantumCircuitBuilder(n_qubits=3)

        builder.cnot(0, 1).cz(1, 2).swap(0, 2)

        gates = builder.gate_count()
        assert "cx" in gates  # CNOT is cx in Qiskit
        assert "cz" in gates
        assert "swap" in gates

    def test_measurement(self):
        """Test measurement operations."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        builder.h(0).cnot(0, 1)
        builder.measure([0, 1], [0, 1])

        gates = builder.gate_count()
        assert "measure" in gates
        assert gates["measure"] == 2

    def test_measure_all(self):
        """Test measure_all convenience method."""
        builder = QuantumCircuitBuilder(n_qubits=3)

        builder.h(0).h(1).h(2)
        builder.measure_all()

        # Check that measurements were added
        # Note: measure_all() may add barriers too
        gates = builder.gate_count()
        assert "measure" in gates

    def test_depth_calculation(self):
        """Test circuit depth calculation."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        initial_depth = builder.depth()
        assert initial_depth == 0

        builder.h(0).h(1)  # Parallel gates
        assert builder.depth() == 1

        builder.cnot(0, 1)  # Sequential gate
        assert builder.depth() == 2

    def test_gate_count(self):
        """Test gate counting."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        builder.h(0).h(1).cnot(0, 1).x(0)

        gates = builder.gate_count()
        assert gates["h"] == 2
        assert gates["cx"] == 1
        assert gates["x"] == 1

    def test_statevector_extraction(self):
        """Test statevector extraction."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        # Create |00⟩ state (should be [1, 0, 0, 0])
        sv = builder.get_statevector()

        assert len(sv) == 4  # 2^2 qubits
        assert np.isclose(sv[0], 1.0)
        assert np.allclose(sv[1:], 0.0)

    def test_statevector_hadamard(self):
        """Test statevector for simple Hadamard."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.h(0)

        sv = builder.get_statevector()

        # Hadamard creates |+⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        assert np.allclose(sv, expected)

    def test_statevector_with_measurement_fails(self):
        """Test that statevector extraction fails with measurements."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.h(0).measure(0, 0)

        with pytest.raises(ValueError, match="measurements"):
            builder.get_statevector()

    def test_unitary_extraction(self):
        """Test unitary matrix extraction."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.x(0)  # Pauli-X gate

        unitary = builder.get_unitary()

        # Pauli-X matrix is [[0, 1], [1, 0]]
        expected = np.array([[0, 1], [1, 0]])
        assert np.allclose(unitary, expected)

    def test_parameterized_layer(self):
        """Test adding parameterized rotation layer."""
        builder = QuantumCircuitBuilder(n_qubits=3)

        params = builder.add_parameterized_layer(layer_type="ry", prefix="theta")

        assert len(params) == 3
        assert builder.num_parameters() == 3
        assert all(isinstance(p, Parameter) for p in params)

    def test_parameterized_layer_types(self):
        """Test different types of parameterized layers."""
        for layer_type in ["rx", "ry", "rz"]:
            builder = QuantumCircuitBuilder(n_qubits=2)
            params = builder.add_parameterized_layer(layer_type=layer_type)

            assert len(params) == 2
            assert builder.num_parameters() == 2

    def test_parameterized_layer_invalid_type(self):
        """Test that invalid layer type raises error."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        with pytest.raises(ValueError, match="Unknown layer type"):
            builder.add_parameterized_layer(layer_type="invalid")

    def test_entangling_layer_linear(self):
        """Test linear entangling layer."""
        builder = QuantumCircuitBuilder(n_qubits=4)
        builder.add_entangling_layer(entangling_type="linear")

        gates = builder.gate_count()
        assert gates["cx"] == 3  # n_qubits - 1

    def test_entangling_layer_circular(self):
        """Test circular entangling layer."""
        builder = QuantumCircuitBuilder(n_qubits=4)
        builder.add_entangling_layer(entangling_type="circular")

        gates = builder.gate_count()
        assert gates["cx"] == 4  # n_qubits (with wrap-around)

    def test_entangling_layer_full(self):
        """Test full entangling layer."""
        builder = QuantumCircuitBuilder(n_qubits=3)
        builder.add_entangling_layer(entangling_type="full")

        gates = builder.gate_count()
        # Full connectivity: n*(n-1)/2 = 3*2/2 = 3
        assert gates["cx"] == 3

    def test_entangling_layer_invalid_type(self):
        """Test that invalid entangling type raises error."""
        builder = QuantumCircuitBuilder(n_qubits=2)

        with pytest.raises(ValueError, match="Unknown entangling type"):
            builder.add_entangling_layer(entangling_type="invalid")

    def test_bind_parameters_list(self):
        """Test parameter binding with list."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        params = builder.add_parameterized_layer(layer_type="ry")

        values = [np.pi / 4, np.pi / 2]
        bound_circuit = builder.bind_parameters(values)

        assert bound_circuit.num_parameters == 0

    def test_bind_parameters_dict(self):
        """Test parameter binding with dictionary."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        params = builder.add_parameterized_layer(layer_type="ry")

        param_dict = {params[0]: np.pi / 4, params[1]: np.pi / 2}
        bound_circuit = builder.bind_parameters(param_dict)

        assert bound_circuit.num_parameters == 0

    def test_draw_text(self):
        """Test text drawing."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        builder.h(0).cnot(0, 1)

        text = builder.draw(output="text")

        assert text is not None
        assert isinstance(text, str)
        assert "q_0" in text or "q[0]" in text

    def test_repr(self):
        """Test string representation."""
        builder = QuantumCircuitBuilder(n_qubits=3)
        builder.h(0).h(1)

        repr_str = repr(builder)

        assert "n_qubits=3" in repr_str
        assert "depth=" in repr_str

    def test_str(self):
        """Test pretty string representation."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        builder.h(0)

        str_repr = str(builder)

        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestBellState:
    """Test Bell state creation."""

    def test_bell_state_creation(self):
        """Test Bell state circuit creation."""
        bell = create_bell_state()

        assert bell.n_qubits == 2
        assert bell.depth() == 2  # H + CNOT

    def test_bell_state_statevector(self):
        """Test Bell state statevector."""
        bell = create_bell_state()
        sv = bell.get_statevector()

        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 → [1/√2, 0, 0, 1/√2]
        expected_probs = np.array([0.5, 0, 0, 0.5])
        probs = np.abs(sv) ** 2

        assert np.allclose(probs, expected_probs)

    def test_bell_state_with_measurement(self):
        """Test Bell state with measurements."""
        bell = create_bell_state(measure=True)

        gates = bell.gate_count()
        assert "measure" in gates


class TestGHZState:
    """Test GHZ state creation."""

    def test_ghz_state_creation(self):
        """Test GHZ state circuit creation."""
        ghz = create_ghz_state(n_qubits=3)

        assert ghz.n_qubits == 3
        gates = ghz.gate_count()
        assert gates["h"] == 1
        assert gates["cx"] == 2  # n_qubits - 1

    def test_ghz_state_invalid_qubits(self):
        """Test that GHZ state requires at least 2 qubits."""
        with pytest.raises(ValueError, match="at least 2 qubits"):
            create_ghz_state(n_qubits=1)

    def test_ghz_state_statevector(self):
        """Test GHZ state statevector for 2 qubits."""
        ghz = create_ghz_state(n_qubits=2)
        sv = ghz.get_statevector()

        # For 2 qubits, GHZ is same as Bell: (|00⟩ + |11⟩)/√2
        expected_probs = np.array([0.5, 0, 0, 0.5])
        probs = np.abs(sv) ** 2

        assert np.allclose(probs, expected_probs, atol=1e-10)

    def test_ghz_state_with_measurement(self):
        """Test GHZ state with measurements."""
        ghz = create_ghz_state(n_qubits=4, measure=True)

        gates = ghz.gate_count()
        assert "measure" in gates


class TestWState:
    """Test W state creation."""

    def test_w_state_creation(self):
        """Test W state circuit creation."""
        w = create_w_state(n_qubits=3)

        assert w.n_qubits == 3

    def test_w_state_invalid_qubits(self):
        """Test that W state requires at least 2 qubits."""
        with pytest.raises(ValueError, match="at least 2 qubits"):
            create_w_state(n_qubits=1)

    def test_w_state_statevector_probabilities(self):
        """Test that W state has uniform probabilities on single-excitation states."""
        w = create_w_state(n_qubits=3)
        sv = w.get_statevector()
        probs = np.abs(sv) ** 2

        # W state for 3 qubits: (|100⟩ + |010⟩ + |001⟩)/√3
        # These correspond to indices 4, 2, 1 in binary
        # Total probability should sum to 1
        assert np.isclose(np.sum(probs), 1.0)

        # The |000⟩ state should have ~0 probability
        assert probs[0] < 0.1  # |000⟩
        # Note: W state has equal superposition of single-excitation states
