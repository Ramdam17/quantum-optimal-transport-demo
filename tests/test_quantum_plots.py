"""
Tests for quantum visualization module.

This module tests plotting functions for quantum circuits, convergence,
cost landscapes, and comparisons.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.visualization.quantum_plots import (plot_convergence,
                                             plot_cost_landscape,
                                             plot_multiple_convergences,
                                             plot_probability_distribution,
                                             plot_quantum_vs_classical,
                                             plot_statevector_bars)


class TestPlotConvergence:
    """Test convergence plotting."""

    def test_basic_convergence(self):
        """Test basic convergence plot."""
        history = [10.0, 8.0, 6.0, 5.0, 4.5]

        fig = plot_convergence(history, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_with_numpy_array(self):
        """Test with numpy array input."""
        history = np.array([10.0, 8.0, 6.0, 5.0, 4.5])

        fig = plot_convergence(history, show=False)

        assert isinstance(fig, Figure)

    def test_custom_labels(self):
        """Test with custom labels."""
        history = [5.0, 4.0, 3.5]

        fig = plot_convergence(
            history, show=False, title="Custom Title", xlabel="Step", ylabel="Loss"
        )

        assert fig.axes[0].get_title() == "Custom Title"
        assert fig.axes[0].get_xlabel() == "Step"
        assert fig.axes[0].get_ylabel() == "Loss"

    def test_log_scale(self):
        """Test logarithmic scale."""
        history = [100.0, 10.0, 1.0, 0.1]

        fig = plot_convergence(history, show=False, log_scale=True)

        assert fig.axes[0].get_yscale() == "log"

    def test_empty_history(self):
        """Test with empty history."""
        history = []

        # Should handle gracefully
        fig = plot_convergence(history, show=False)
        assert isinstance(fig, Figure)


class TestPlotCostLandscape:
    """Test cost landscape plotting."""

    def test_simple_landscape(self):
        """Test simple quadratic cost landscape."""

        def cost_fn(params):
            return (params[0] - 1) ** 2 + (params[1] + 2) ** 2

        ranges = [(0, 2), (-3, -1)]

        fig = plot_cost_landscape(ranges, cost_fn, show=False, resolution=10)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Main axis + colorbar

    def test_with_optimal_point(self):
        """Test marking optimal point."""

        def cost_fn(params):
            return params[0] ** 2 + params[1] ** 2

        ranges = [(-2, 2), (-2, 2)]
        optimal = np.array([0.5, 0.5])

        fig = plot_cost_landscape(
            ranges, cost_fn, optimal_params=optimal, show=False, resolution=10
        )

        assert isinstance(fig, Figure)

    def test_custom_title(self):
        """Test with custom title."""

        def cost_fn(params):
            return params[0] ** 2 + params[1] ** 2

        ranges = [(-1, 1), (-1, 1)]

        fig = plot_cost_landscape(
            ranges, cost_fn, show=False, resolution=5, title="Energy Landscape"
        )

        assert "Energy Landscape" in fig.axes[0].get_title()


class TestPlotQuantumVsClassical:
    """Test quantum vs classical comparison plotting."""

    def test_basic_comparison(self):
        """Test basic comparison plot."""
        classical = {"cost": 2.5}
        quantum = {"cost": 2.8, "convergence_history": [10.0, 8.0, 5.0, 3.0, 2.8]}

        fig = plot_quantum_vs_classical(classical, quantum, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Two subplots

    def test_without_convergence(self):
        """Test without convergence history."""
        classical = {"cost": 2.5}
        quantum = {"cost": 2.8}

        fig = plot_quantum_vs_classical(classical, quantum, show=False)

        assert isinstance(fig, Figure)

    def test_quantum_better(self):
        """Test when quantum cost is lower."""
        classical = {"cost": 3.0}
        quantum = {"cost": 2.5, "convergence_history": [10.0, 5.0, 2.5]}

        fig = plot_quantum_vs_classical(classical, quantum, show=False)

        assert isinstance(fig, Figure)


class TestPlotStatevectorBars:
    """Test statevector bar plotting."""

    def test_simple_state(self):
        """Test simple two-qubit state."""
        state = np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j])

        fig = plot_statevector_bars(state, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Amplitude and phase

    def test_with_phases(self):
        """Test state with complex phases."""
        state = np.array(
            [
                0.5 + 0j,
                0.5 * np.exp(1j * np.pi / 4),
                0.5 * np.exp(1j * np.pi / 2),
                0.5 * np.exp(1j * 3 * np.pi / 4),
            ]
        )

        fig = plot_statevector_bars(state, show=False)

        assert isinstance(fig, Figure)

    def test_max_states_limit(self):
        """Test limiting number of displayed states."""
        state = np.random.randn(32) + 1j * np.random.randn(32)
        state = state / np.linalg.norm(state)

        fig = plot_statevector_bars(state, show=False, max_states=8)

        assert isinstance(fig, Figure)

    def test_custom_title(self):
        """Test with custom title."""
        state = np.array([0.7 + 0j, 0.3 + 0j, 0, 0])

        fig = plot_statevector_bars(state, show=False, title="Bell State")

        assert isinstance(fig, Figure)


class TestPlotProbabilityDistribution:
    """Test probability distribution plotting."""

    def test_dict_input(self):
        """Test with dictionary input."""
        probs = {"00": 0.5, "01": 0.3, "10": 0.2}

        fig = plot_probability_distribution(probs, show=False)

        assert isinstance(fig, Figure)

    def test_array_input(self):
        """Test with array input."""
        probs = np.array([0.5, 0.3, 0.2, 0.0])

        fig = plot_probability_distribution(probs, show=False)

        assert isinstance(fig, Figure)

    def test_threshold_filtering(self):
        """Test filtering low probabilities."""
        probs = {
            "00": 0.5,
            "01": 0.3,
            "10": 0.15,
            "11": 0.05,
            "100": 0.001,  # Below threshold
        }

        fig = plot_probability_distribution(probs, show=False, threshold=0.01)

        assert isinstance(fig, Figure)

    def test_max_states_limit(self):
        """Test limiting displayed states."""
        probs = {f"{i:03b}": 0.1 for i in range(20)}

        fig = plot_probability_distribution(probs, show=False, max_states=8)

        assert isinstance(fig, Figure)

    def test_custom_title(self):
        """Test with custom title."""
        probs = {"0": 0.6, "1": 0.4}

        fig = plot_probability_distribution(
            probs, show=False, title="Measurement Results"
        )

        assert fig.axes[0].get_title() == "Measurement Results"


class TestPlotMultipleConvergences:
    """Test multiple convergence comparison."""

    def test_two_methods(self):
        """Test comparing two methods."""
        histories = {
            "VQE": np.array([10.0, 8.0, 6.0, 5.0]),
            "QAOA": np.array([12.0, 9.0, 7.0, 6.5]),
        }

        fig = plot_multiple_convergences(histories, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_multiple_methods(self):
        """Test comparing many methods."""
        histories = {
            "Method A": np.array([10, 8, 6]),
            "Method B": np.array([12, 9, 7]),
            "Method C": np.array([11, 8.5, 6.5]),
            "Method D": np.array([13, 10, 8]),
        }

        fig = plot_multiple_convergences(histories, show=False)

        assert isinstance(fig, Figure)

    def test_different_lengths(self):
        """Test with different history lengths."""
        histories = {"Short": np.array([10, 5]), "Long": np.array([12, 10, 8, 6, 5, 4])}

        fig = plot_multiple_convergences(histories, show=False)

        assert isinstance(fig, Figure)

    def test_log_scale(self):
        """Test with logarithmic scale."""
        histories = {
            "Method A": np.array([100, 10, 1, 0.1]),
            "Method B": np.array([200, 20, 2, 0.2]),
        }

        fig = plot_multiple_convergences(histories, show=False, log_scale=True)

        assert fig.axes[0].get_yscale() == "log"

    def test_custom_title(self):
        """Test with custom title."""
        histories = {"A": np.array([10, 5]), "B": np.array([12, 6])}

        fig = plot_multiple_convergences(
            histories, show=False, title="Algorithm Comparison"
        )

        assert fig.axes[0].get_title() == "Algorithm Comparison"


class TestIntegration:
    """Integration tests combining multiple plots."""

    def test_full_visualization_suite(self):
        """Test creating all plot types."""
        # Convergence
        history = [10.0, 8.0, 6.0, 5.0, 4.5]
        fig1 = plot_convergence(history, show=False)
        assert isinstance(fig1, Figure)

        # Cost landscape
        def cost_fn(params):
            return params[0] ** 2 + params[1] ** 2

        ranges = [(-2, 2), (-2, 2)]
        fig2 = plot_cost_landscape(ranges, cost_fn, show=False, resolution=5)
        assert isinstance(fig2, Figure)

        # Quantum vs classical
        classical = {"cost": 2.5}
        quantum = {"cost": 2.8, "convergence_history": history}
        fig3 = plot_quantum_vs_classical(classical, quantum, show=False)
        assert isinstance(fig3, Figure)

        # Statevector
        state = np.array([0.7 + 0j, 0.3 + 0j, 0, 0])
        fig4 = plot_statevector_bars(state, show=False)
        assert isinstance(fig4, Figure)

        # Probabilities
        probs = {"00": 0.5, "01": 0.3, "10": 0.2}
        fig5 = plot_probability_distribution(probs, show=False)
        assert isinstance(fig5, Figure)

        # Multiple convergences
        histories = {"VQE": np.array(history), "QAOA": np.array([12, 9, 7, 6, 5.5])}
        fig6 = plot_multiple_convergences(histories, show=False)
        assert isinstance(fig6, Figure)

    def test_realistic_qot_workflow(self):
        """Test realistic QOT visualization workflow."""
        # Simulate QOT results
        classical_result = {"cost": 2.5, "iterations": 100}

        quantum_result = {
            "cost": 2.8,
            "optimal_params": np.array([1.2, 0.8, 1.5, 0.3]),
            "final_state": np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j]),
            "final_probabilities": {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25},
            "iterations": 50,
            "convergence_history": [10.0, 8.0, 6.0, 4.5, 3.5, 3.0, 2.8],
        }

        # Comparison plot
        fig1 = plot_quantum_vs_classical(classical_result, quantum_result, show=False)
        assert isinstance(fig1, Figure)

        # Final state visualization
        fig2 = plot_statevector_bars(quantum_result["final_state"], show=False)
        assert isinstance(fig2, Figure)

        # Probability distribution
        fig3 = plot_probability_distribution(
            quantum_result["final_probabilities"], show=False
        )
        assert isinstance(fig3, Figure)
