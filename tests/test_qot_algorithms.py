"""
Tests for quantum optimal transport algorithms module.

This module tests the quantum OT implementations including VQE and QAOA
approaches for computing optimal transport between probability distributions.
"""

import numpy as np
import pytest

from src.quantum.qot_algorithms import QuantumOT


class TestQuantumOTInitialization:
    """Test QuantumOT initialization and validation."""

    def test_valid_initialization(self):
        """Test valid initialization."""
        qot = QuantumOT(n_qubits=4, method="vqe")

        assert qot.n_qubits == 4
        assert qot.method == "vqe"
        assert qot.max_iterations == 100
        assert qot.optimizer_name == "COBYLA"
        assert qot.shots == 1024
        assert qot.seed is None

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        qot = QuantumOT(
            n_qubits=6,
            method="qaoa",
            max_iterations=50,
            optimizer="SLSQP",
            shots=2048,
            seed=42,
        )

        assert qot.n_qubits == 6
        assert qot.method == "qaoa"
        assert qot.max_iterations == 50
        assert qot.optimizer_name == "SLSQP"
        assert qot.shots == 2048
        assert qot.seed == 42

    def test_invalid_n_qubits(self):
        """Test error on invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be at least 2"):
            QuantumOT(n_qubits=1)

    def test_invalid_method(self):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            QuantumOT(n_qubits=4, method="invalid")

    def test_invalid_optimizer(self):
        """Test error on invalid optimizer."""
        with pytest.raises(ValueError, match="Invalid optimizer"):
            QuantumOT(n_qubits=4, optimizer="SGD")

    def test_repr(self):
        """Test string representation."""
        qot = QuantumOT(n_qubits=4, method="vqe", optimizer="COBYLA")
        repr_str = repr(qot)

        assert "QuantumOT" in repr_str
        assert "n_qubits=4" in repr_str
        assert "method='vqe'" in repr_str
        assert "optimizer='COBYLA'" in repr_str


class TestQuantumOTCompute:
    """Test quantum OT computation."""

    def test_simple_two_point_vqe(self):
        """Test VQE on simple two-point distribution."""
        qot = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)

        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        result = qot.compute(source, target)

        # Check result structure
        assert "cost" in result
        assert "optimal_params" in result
        assert "final_state" in result
        assert "final_probabilities" in result
        assert "iterations" in result
        assert "convergence_history" in result
        assert "execution_time" in result
        assert "success" in result

        # Check values
        assert result["cost"] >= 0
        assert result["iterations"] > 0
        assert result["execution_time"] > 0
        assert len(result["convergence_history"]) == result["iterations"]

    def test_simple_two_point_qaoa(self):
        """Test QAOA on simple two-point distribution."""
        qot = QuantumOT(n_qubits=2, method="qaoa", max_iterations=20, seed=42)

        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        result = qot.compute(source, target)

        # Check result structure
        assert "cost" in result
        assert "optimal_params" in result
        assert "final_state" in result
        assert "iterations" in result

        # QAOA should have fewer parameters than VQE
        assert len(result["optimal_params"]) == 4  # 2 layers × 2 params

    def test_four_point_distribution(self):
        """Test on four-point distribution."""
        qot = QuantumOT(n_qubits=3, method="vqe", max_iterations=30, seed=42)

        source = np.array([0.1, 0.2, 0.3, 0.4])
        target = np.array([0.4, 0.3, 0.2, 0.1])

        result = qot.compute(source, target)

        assert result["cost"] >= 0
        assert result["iterations"] > 0
        assert len(result["final_probabilities"]) > 0

    def test_uniform_distributions(self):
        """Test on uniform distributions (should have low cost)."""
        qot = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)

        uniform = np.array([0.5, 0.5])

        result = qot.compute(uniform, uniform)

        # Cost should be relatively low for identical distributions
        assert result["cost"] >= 0
        assert result["cost"] < 10.0  # Reasonable upper bound

    def test_custom_cost_matrix(self):
        """Test with custom cost matrix."""
        qot = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)

        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        # Custom cost matrix (high cost for diagonal, low for off-diagonal)
        cost_matrix = np.array([[10.0, 1.0], [1.0, 10.0]])

        result = qot.compute(source, target, cost_matrix=cost_matrix)

        assert result["cost"] >= 0

    def test_convergence_history(self):
        """Test that convergence history is tracked."""
        qot = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)

        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        result = qot.compute(source, target)

        history = result["convergence_history"]

        # History should have same length as iterations
        assert len(history) == result["iterations"]

        # All costs should be non-negative
        assert all(cost >= 0 for cost in history)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with same seed."""
        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        qot1 = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)
        result1 = qot1.compute(source, target)

        qot2 = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)
        result2 = qot2.compute(source, target)

        # Should produce same results
        np.testing.assert_allclose(result1["cost"], result2["cost"], rtol=1e-5)
        np.testing.assert_allclose(
            result1["optimal_params"], result2["optimal_params"], rtol=1e-5
        )


class TestQuantumOTValidation:
    """Test input validation."""

    def test_invalid_source_dimension(self):
        """Test error on 2D source."""
        qot = QuantumOT(n_qubits=2, method="vqe")

        source = np.array([[0.3, 0.7]])  # 2D
        target = np.array([0.6, 0.4])

        with pytest.raises(ValueError, match="must be 1D arrays"):
            qot.compute(source, target)

    def test_invalid_target_dimension(self):
        """Test error on 2D target."""
        qot = QuantumOT(n_qubits=2, method="vqe")

        source = np.array([0.3, 0.7])
        target = np.array([[0.6, 0.4]])  # 2D

        with pytest.raises(ValueError, match="must be 1D arrays"):
            qot.compute(source, target)

    def test_mismatched_lengths(self):
        """Test error on mismatched distribution lengths."""
        qot = QuantumOT(n_qubits=2, method="vqe")

        source = np.array([0.3, 0.7])
        target = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError, match="must have same length"):
            qot.compute(source, target)

    def test_automatic_normalization(self):
        """Test that distributions are automatically normalized."""
        qot = QuantumOT(n_qubits=2, method="vqe", max_iterations=20, seed=42)

        # Non-normalized distributions
        source = np.array([3.0, 7.0])
        target = np.array([6.0, 4.0])

        result = qot.compute(source, target)

        # Should work (normalized internally)
        assert result["cost"] >= 0


class TestQuantumOTCircuitBuilding:
    """Test internal circuit building methods."""

    def test_vqe_circuit_building(self):
        """Test VQE circuit construction."""
        qot = QuantumOT(n_qubits=3, method="vqe", seed=42)

        # Build VQE circuit
        n_layers = 2
        n_params = n_layers * qot.n_qubits * 2
        params = np.random.randn(n_params)

        circuit = qot._build_vqe_circuit(params, n_layers)

        # Check circuit is valid
        assert circuit.n_qubits == qot.n_qubits
        assert circuit.depth() > 0

    def test_qaoa_circuit_building(self):
        """Test QAOA circuit construction."""
        qot = QuantumOT(n_qubits=3, method="qaoa", seed=42)

        # Build QAOA circuit
        p = 2
        n_params = 2 * p
        params = np.random.randn(n_params)
        cost_matrix = qot._default_cost_matrix(4)

        circuit = qot._build_qaoa_circuit(params, p, cost_matrix)

        # Check circuit is valid
        assert circuit.n_qubits == qot.n_qubits
        assert circuit.depth() > 0

    def test_default_cost_matrix(self):
        """Test default cost matrix creation."""
        qot = QuantumOT(n_qubits=2, method="vqe")

        cost_matrix = qot._default_cost_matrix(4)

        # Check properties
        assert cost_matrix.shape == (4, 4)
        assert np.allclose(cost_matrix, cost_matrix.T)  # Symmetric
        assert np.all(np.diag(cost_matrix) == 0)  # Zero diagonal


class TestQuantumOTOptimizers:
    """Test different optimizers."""

    @pytest.mark.parametrize("optimizer", ["COBYLA", "SLSQP", "Nelder-Mead"])
    def test_different_optimizers(self, optimizer):
        """Test different classical optimizers."""
        qot = QuantumOT(
            n_qubits=2, method="vqe", optimizer=optimizer, max_iterations=20, seed=42
        )

        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        result = qot.compute(source, target)

        # Should complete successfully
        assert result["cost"] >= 0
        assert result["iterations"] > 0


class TestQuantumOTIntegration:
    """Integration tests with full pipeline."""

    def test_vqe_vs_qaoa_comparison(self):
        """Compare VQE and QAOA results."""
        source = np.array([0.3, 0.7])
        target = np.array([0.6, 0.4])

        # VQE
        qot_vqe = QuantumOT(n_qubits=2, method="vqe", max_iterations=30, seed=42)
        result_vqe = qot_vqe.compute(source, target)

        # QAOA
        qot_qaoa = QuantumOT(n_qubits=2, method="qaoa", max_iterations=30, seed=42)
        result_qaoa = qot_qaoa.compute(source, target)

        # Both should produce valid results
        assert result_vqe["cost"] >= 0
        assert result_qaoa["cost"] >= 0

    def test_larger_distribution(self):
        """Test with larger distribution (8 points)."""
        qot = QuantumOT(n_qubits=4, method="vqe", max_iterations=30, seed=42)

        source = np.array([0.1, 0.15, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1])
        target = np.array([0.15, 0.1, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1])

        result = qot.compute(source, target)

        assert result["cost"] >= 0
        assert result["iterations"] > 0

    def test_extreme_distributions(self):
        """Test with extreme distributions (one peak)."""
        qot = QuantumOT(n_qubits=3, method="vqe", max_iterations=30, seed=42)

        # Concentrated source, spread target
        source = np.array([0.9, 0.05, 0.05, 0.0])
        target = np.array([0.25, 0.25, 0.25, 0.25])

        result = qot.compute(source, target)

        # Should handle extreme cases
        assert result["cost"] >= 0
        assert result["cost"] < np.inf
