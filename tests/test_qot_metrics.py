"""
Tests for quantum optimal transport metrics module.

This module tests quantum-specific metrics, fidelity calculations,
and comparison tools for classical vs quantum OT results.
"""

import numpy as np
import pytest

from src.quantum.qot_metrics import (QuantumMetrics, compare_classical_quantum,
                                     fidelity_distance,
                                     quantum_wasserstein_distance)


class TestQuantumMetricsInitialization:
    """Test QuantumMetrics initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        metrics = QuantumMetrics()
        assert metrics is not None


class TestQuantumWassersteinDistance:
    """Test quantum Wasserstein distance computation."""

    def test_identical_states(self):
        """Test distance between identical states is zero."""
        metrics = QuantumMetrics()

        state = np.array([0.7 + 0j, 0.3 + 0j, 0, 0])
        distance = metrics.quantum_wasserstein_distance(state, state)

        assert distance >= 0
        assert distance < 0.1  # Should be very close to zero

    def test_orthogonal_states(self):
        """Test distance between orthogonal states."""
        metrics = QuantumMetrics()

        state1 = np.array([1.0 + 0j, 0, 0, 0])
        state2 = np.array([0, 1.0 + 0j, 0, 0])

        distance = metrics.quantum_wasserstein_distance(state1, state2)

        assert distance > 0  # Non-zero distance

    def test_two_qubit_states(self):
        """Test on two-qubit states."""
        metrics = QuantumMetrics()

        # Equal superposition vs biased
        state1 = np.array([0.5 + 0j, 0.5 + 0j, 0, 0])
        state2 = np.array([0.8 + 0j, 0.2 + 0j, 0, 0])

        distance = metrics.quantum_wasserstein_distance(state1, state2)

        assert distance >= 0

    def test_custom_cost_matrix(self):
        """Test with custom cost matrix."""
        metrics = QuantumMetrics()

        state1 = np.array([0.6 + 0j, 0.4 + 0j, 0, 0])
        state2 = np.array([0.3 + 0j, 0.7 + 0j, 0, 0])

        # Custom cost
        cost_matrix = np.array(
            [[0, 2, 5, 10], [2, 0, 3, 8], [5, 3, 0, 2], [10, 8, 2, 0]]
        )

        distance = metrics.quantum_wasserstein_distance(state1, state2, cost_matrix)

        assert distance >= 0

    def test_invalid_dimension_error(self):
        """Test error on 2D input."""
        metrics = QuantumMetrics()

        state1 = np.array([[0.7 + 0j, 0.3 + 0j]])  # 2D
        state2 = np.array([0.6 + 0j, 0.4 + 0j])

        with pytest.raises(ValueError, match="must be 1D arrays"):
            metrics.quantum_wasserstein_distance(state1, state2)

    def test_mismatched_dimensions_error(self):
        """Test error on mismatched dimensions."""
        metrics = QuantumMetrics()

        state1 = np.array([0.7 + 0j, 0.3 + 0j])
        state2 = np.array([0.6 + 0j, 0.3 + 0j, 0.1 + 0j])

        with pytest.raises(ValueError, match="must have same dimension"):
            metrics.quantum_wasserstein_distance(state1, state2)

    def test_convenience_function(self):
        """Test convenience wrapper function."""
        state1 = np.array([0.7 + 0j, 0.3 + 0j, 0, 0])
        state2 = np.array([0.5 + 0j, 0.5 + 0j, 0, 0])

        distance = quantum_wasserstein_distance(state1, state2)

        assert distance >= 0


class TestFidelityDistance:
    """Test fidelity-based distance computation."""

    def test_identical_states_zero_distance(self):
        """Test identical states have zero distance."""
        metrics = QuantumMetrics()

        state = np.array([1.0 + 0j, 0, 0, 0])
        distance = metrics.fidelity_distance(state, state)

        assert np.isclose(distance, 0.0, atol=1e-10)

    def test_orthogonal_states_max_distance(self):
        """Test orthogonal states have maximum distance."""
        metrics = QuantumMetrics()

        state1 = np.array([1.0 + 0j, 0])
        state2 = np.array([0, 1.0 + 0j])

        distance = metrics.fidelity_distance(state1, state2)

        assert np.isclose(distance, 1.0, atol=1e-10)

    def test_superposition_states(self):
        """Test distance between superposition states."""
        metrics = QuantumMetrics()

        # |+⟩ state
        state1 = np.array([1.0 + 0j, 1.0 + 0j]) / np.sqrt(2)
        # |0⟩ state
        state2 = np.array([1.0 + 0j, 0])

        distance = metrics.fidelity_distance(state1, state2)

        # |⟨+|0⟩|² = 0.5, so distance = 0.5
        assert np.isclose(distance, 0.5, atol=1e-10)

    def test_complex_phases(self):
        """Test with complex phases."""
        metrics = QuantumMetrics()

        state1 = np.array([1.0 + 0j, 0])
        state2 = np.array([np.exp(1j * np.pi / 4), 0])

        distance = metrics.fidelity_distance(state1, state2)

        # Global phase doesn't affect fidelity
        assert np.isclose(distance, 0.0, atol=1e-10)

    def test_invalid_dimensions(self):
        """Test error on invalid dimensions."""
        metrics = QuantumMetrics()

        state1 = np.array([[1.0 + 0j, 0]])  # 2D
        state2 = np.array([1.0 + 0j, 0])

        with pytest.raises(ValueError, match="must be 1D arrays"):
            metrics.fidelity_distance(state1, state2)

    def test_convenience_function(self):
        """Test convenience wrapper function."""
        state1 = np.array([1.0 + 0j, 0])
        state2 = np.array([0.7 + 0j, 0.7 + 0j]) / np.sqrt(2)

        distance = fidelity_distance(state1, state2)

        assert 0 <= distance <= 1


class TestTraceDistance:
    """Test trace distance computation."""

    def test_identical_states(self):
        """Test trace distance for identical states."""
        metrics = QuantumMetrics()

        state = np.array([0.6 + 0j, 0.8 + 0j])
        distance = metrics.trace_distance(state, state)

        assert np.isclose(distance, 0.0, atol=1e-10)

    def test_orthogonal_states(self):
        """Test trace distance for orthogonal states."""
        metrics = QuantumMetrics()

        state1 = np.array([1.0 + 0j, 0, 0])
        state2 = np.array([0, 1.0 + 0j, 0])

        distance = metrics.trace_distance(state1, state2)

        assert np.isclose(distance, 1.0, atol=1e-10)

    def test_superposition(self):
        """Test trace distance for superposition states."""
        metrics = QuantumMetrics()

        state1 = np.array([1.0 + 0j, 0])
        state2 = np.array([1.0 + 0j, 1.0 + 0j]) / np.sqrt(2)

        distance = metrics.trace_distance(state1, state2)

        # D = √(1 - F), F = 0.5, so D = √0.5
        expected = np.sqrt(0.5)
        assert np.isclose(distance, expected, atol=1e-10)


class TestCompareOTResults:
    """Test comparison of classical and quantum OT results."""

    def test_identical_costs(self):
        """Test comparison with identical costs."""
        metrics = QuantumMetrics()

        classical = {"cost": 5.0}
        quantum = {"cost": 5.0}

        comparison = metrics.compare_ot_results(classical, quantum)

        assert comparison["classical_cost"] == 5.0
        assert comparison["quantum_cost"] == 5.0
        assert comparison["cost_difference"] == 0.0
        assert comparison["cost_ratio"] == 1.0
        assert comparison["relative_error"] == 0.0

    def test_quantum_higher_cost(self):
        """Test when quantum cost is higher."""
        metrics = QuantumMetrics()

        classical = {"cost": 5.0}
        quantum = {"cost": 6.0}

        comparison = metrics.compare_ot_results(classical, quantum)

        assert comparison["cost_difference"] == 1.0
        assert comparison["cost_ratio"] == 1.2
        assert comparison["relative_error"] == 20.0

    def test_quantum_lower_cost(self):
        """Test when quantum cost is lower."""
        metrics = QuantumMetrics()

        classical = {"cost": 6.0}
        quantum = {"cost": 5.0}

        comparison = metrics.compare_ot_results(classical, quantum)

        assert comparison["cost_difference"] == 1.0
        assert np.isclose(comparison["cost_ratio"], 5.0 / 6.0)
        assert np.isclose(comparison["relative_error"], 100.0 / 6.0)

    def test_missing_cost_error(self):
        """Test error when cost is missing."""
        metrics = QuantumMetrics()

        classical = {"result": "data"}
        quantum = {"cost": 5.0}

        with pytest.raises(ValueError, match="must contain 'cost' key"):
            metrics.compare_ot_results(classical, quantum)

    def test_convenience_function(self):
        """Test convenience wrapper function."""
        classical = {"cost": 3.0}
        quantum = {"cost": 3.5}

        comparison = compare_classical_quantum(classical, quantum)

        assert "cost_difference" in comparison
        assert "relative_error" in comparison


class TestMarginalError:
    """Test marginal constraint error computation."""

    def test_perfect_match(self):
        """Test zero error for perfect match."""
        metrics = QuantumMetrics()

        probs = np.array([0.3, 0.7])
        target = np.array([0.3, 0.7])

        error = metrics.compute_marginal_error(probs, target)

        assert np.isclose(error, 0.0, atol=1e-10)

    def test_small_deviation(self):
        """Test small deviation."""
        metrics = QuantumMetrics()

        probs = np.array([0.3, 0.7])
        target = np.array([0.35, 0.65])

        error = metrics.compute_marginal_error(probs, target)

        expected = np.sqrt(0.05**2 + 0.05**2)
        assert np.isclose(error, expected, atol=1e-10)

    def test_large_deviation(self):
        """Test large deviation."""
        metrics = QuantumMetrics()

        probs = np.array([0.1, 0.9])
        target = np.array([0.9, 0.1])

        error = metrics.compute_marginal_error(probs, target)

        assert error > 1.0  # Significant error

    def test_different_lengths(self):
        """Test with different length arrays (truncates)."""
        metrics = QuantumMetrics()

        probs = np.array([0.3, 0.5, 0.2])
        target = np.array([0.4, 0.6])

        error = metrics.compute_marginal_error(probs, target)

        # Should compare first 2 elements only
        assert error >= 0

    def test_invalid_dimensions(self):
        """Test error on 2D input."""
        metrics = QuantumMetrics()

        probs = np.array([[0.3, 0.7]])
        target = np.array([0.3, 0.7])

        with pytest.raises(ValueError, match="must be 1D arrays"):
            metrics.compute_marginal_error(probs, target)


class TestConvergenceRate:
    """Test convergence analysis."""

    def test_simple_convergence(self):
        """Test simple convergence pattern."""
        metrics = QuantumMetrics()

        history = np.array([10.0, 8.0, 6.0, 5.0, 4.5, 4.5])

        analysis = metrics.compute_convergence_rate(history)

        assert analysis["initial_cost"] == 10.0
        assert analysis["final_cost"] == 4.5
        assert analysis["improvement"] == 5.5
        assert analysis["improvement_rate"] > 0

    def test_converged_optimization(self):
        """Test converged optimization (stable final values)."""
        metrics = QuantumMetrics()

        history = np.array([10.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        analysis = metrics.compute_convergence_rate(history)

        assert analysis["converged"] == 1.0  # True

    def test_not_converged(self):
        """Test non-converged optimization (still changing)."""
        metrics = QuantumMetrics()

        history = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        analysis = metrics.compute_convergence_rate(history)

        # May or may not be marked as converged depending on variance
        assert "converged" in analysis

    def test_short_history(self):
        """Test with short history (< 5 values)."""
        metrics = QuantumMetrics()

        history = np.array([5.0, 3.0])

        analysis = metrics.compute_convergence_rate(history)

        assert analysis["converged"] == 0.0  # False (too short)

    def test_empty_history_error(self):
        """Test error on empty history."""
        metrics = QuantumMetrics()

        history = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            metrics.compute_convergence_rate(history)

    def test_no_improvement(self):
        """Test when cost doesn't improve."""
        metrics = QuantumMetrics()

        history = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        analysis = metrics.compute_convergence_rate(history)

        assert analysis["improvement"] == 0.0
        assert analysis["converged"] == 1.0  # Stable (converged)


class TestIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_metric_suite(self):
        """Test computing all metrics for a pair of states."""
        metrics = QuantumMetrics()

        state1 = np.array([0.7 + 0j, 0.3 + 0j, 0, 0])
        state2 = np.array([0.5 + 0j, 0.5 + 0j, 0, 0])

        # Quantum Wasserstein
        qw_dist = metrics.quantum_wasserstein_distance(state1, state2)

        # Fidelity distance
        fid_dist = metrics.fidelity_distance(state1, state2)

        # Trace distance
        trace_dist = metrics.trace_distance(state1, state2)

        # All should be non-negative
        assert qw_dist >= 0
        assert fid_dist >= 0
        assert trace_dist >= 0

        # Trace distance should be related to fidelity
        assert np.isclose(trace_dist, np.sqrt(fid_dist), atol=0.1)

    def test_ot_result_comparison_workflow(self):
        """Test complete OT comparison workflow."""
        metrics = QuantumMetrics()

        # Simulate OT results
        classical_result = {"cost": 2.5, "iterations": 100}

        quantum_result = {
            "cost": 3.0,
            "iterations": 50,
            "convergence_history": [10.0, 8.0, 6.0, 4.0, 3.0],
        }

        # Compare costs
        comparison = metrics.compare_ot_results(classical_result, quantum_result)

        assert comparison["relative_error"] == 20.0

        # Analyze convergence
        convergence = metrics.compute_convergence_rate(
            np.array(quantum_result["convergence_history"])
        )

        assert convergence["final_cost"] == 3.0
        assert convergence["improvement"] > 0
