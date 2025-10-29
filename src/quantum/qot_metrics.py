"""
Quantum Optimal Transport metrics module.

This module implements metrics for evaluating and comparing quantum optimal
transport results with classical methods. It includes quantum-specific distance
measures and fidelity-based comparisons.

Classes
-------
QuantumMetrics
    Collection of quantum OT metrics and comparison tools

Functions
---------
quantum_wasserstein_distance
    Compute quantum Wasserstein distance between states
fidelity_distance
    Compute fidelity-based distance between quantum states
compare_classical_quantum
    Compare classical and quantum OT results

Examples
--------
>>> from src.quantum.qot_metrics import quantum_wasserstein_distance
>>> import numpy as np
>>>
>>> state1 = np.array([0.5+0j, 0.5+0j, 0, 0])
>>> state2 = np.array([0.3+0j, 0.7+0j, 0, 0])
>>> distance = quantum_wasserstein_distance(state1, state2)

Notes
-----
Quantum Wasserstein distances generalize classical Wasserstein distances
to quantum states. They measure the minimal cost of transforming one
quantum state into another.

References
----------
.. [1] De Palma, G., Trevisan, D. (2021). "Quantum Optimal Transport"
.. [2] Golse, F., et al. (2016). "On the Mean Field and Classical Limits
       of Quantum Mechanics"
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuantumMetrics:
    """
    Quantum optimal transport metrics and comparison tools.

    This class provides methods for computing quantum-specific metrics,
    comparing quantum and classical OT results, and analyzing the
    performance of quantum algorithms.

    Methods
    -------
    quantum_wasserstein_distance
        Compute quantum Wasserstein distance
    fidelity_distance
        Compute fidelity-based distance
    compare_ot_results
        Compare classical and quantum OT costs
    compute_marginal_error
        Compute marginal constraint violation

    Examples
    --------
    >>> metrics = QuantumMetrics()
    >>> state1 = np.array([0.5+0j, 0.5+0j, 0, 0])
    >>> state2 = np.array([0.3+0j, 0.7+0j, 0, 0])
    >>> distance = metrics.quantum_wasserstein_distance(state1, state2)
    """

    def __init__(self):
        """Initialize QuantumMetrics."""
        logger.debug("Initialized QuantumMetrics")

    def quantum_wasserstein_distance(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute quantum Wasserstein distance between quantum states.

        The quantum Wasserstein distance generalizes the classical
        Wasserstein distance to quantum states. For pure states,
        it can be approximated using the fidelity.

        Parameters
        ----------
        state1 : np.ndarray
            First quantum state (statevector)
        state2 : np.ndarray
            Second quantum state (statevector)
        cost_matrix : Optional[np.ndarray], optional
            Cost matrix for transport, by default None

        Returns
        -------
        float
            Quantum Wasserstein distance

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> state1 = np.array([1.0+0j, 0, 0, 0])
        >>> state2 = np.array([0, 1.0+0j, 0, 0])
        >>> distance = metrics.quantum_wasserstein_distance(state1, state2)

        Notes
        -----
        For pure states |ψ⟩ and |φ⟩, the quantum Wasserstein distance
        can be related to the fidelity F(ψ,φ) = |⟨ψ|φ⟩|²
        """
        # Validate inputs
        state1 = np.asarray(state1, dtype=complex)
        state2 = np.asarray(state2, dtype=complex)

        if state1.ndim != 1 or state2.ndim != 1:
            raise ValueError("States must be 1D arrays (statevectors)")

        if len(state1) != len(state2):
            raise ValueError("States must have same dimension")

        # Extract probability distributions from quantum states
        probs1 = np.abs(state1) ** 2
        probs2 = np.abs(state2) ** 2

        # Normalize
        probs1 = probs1 / np.sum(probs1) if np.sum(probs1) > 0 else probs1
        probs2 = probs2 / np.sum(probs2) if np.sum(probs2) > 0 else probs2

        # Default cost matrix (Euclidean distance on computational basis)
        if cost_matrix is None:
            n = len(probs1)
            indices = np.arange(n).reshape(-1, 1)
            cost_matrix = np.abs(indices - indices.T)

        # Compute Wasserstein-1 distance (Earth Mover's Distance)
        # Simplified calculation for 1D distributions
        distance = self._compute_emd_1d(probs1, probs2, cost_matrix)

        return float(distance)

    def fidelity_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute fidelity-based distance between quantum states.

        The fidelity F(ψ,φ) = |⟨ψ|φ⟩|² measures the overlap between
        two quantum states. The fidelity distance is 1 - F(ψ,φ).

        Parameters
        ----------
        state1 : np.ndarray
            First quantum state
        state2 : np.ndarray
            Second quantum state

        Returns
        -------
        float
            Fidelity distance (between 0 and 1)

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> state1 = np.array([1.0+0j, 0])
        >>> state2 = np.array([1.0+0j, 0])
        >>> distance = metrics.fidelity_distance(state1, state2)
        >>> print(distance)  # Should be 0 (identical states)

        Notes
        -----
        Properties:
        - F(ψ,φ) = 1 if states are identical
        - F(ψ,φ) = 0 if states are orthogonal
        - Distance = 1 - F(ψ,φ)
        """
        state1 = np.asarray(state1, dtype=complex)
        state2 = np.asarray(state2, dtype=complex)

        if state1.ndim != 1 or state2.ndim != 1:
            raise ValueError("States must be 1D arrays")

        if len(state1) != len(state2):
            raise ValueError("States must have same dimension")

        # Compute fidelity
        fidelity = np.abs(np.vdot(state1, state2)) ** 2

        # Fidelity distance
        distance = 1.0 - fidelity

        return float(distance)

    def trace_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute trace distance between quantum states.

        For pure states |ψ⟩ and |φ⟩, the trace distance is:
        D(ψ,φ) = √(1 - F(ψ,φ))

        where F is the fidelity.

        Parameters
        ----------
        state1 : np.ndarray
            First quantum state
        state2 : np.ndarray
            Second quantum state

        Returns
        -------
        float
            Trace distance (between 0 and 1)

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> state1 = np.array([1.0+0j, 0])
        >>> state2 = np.array([0, 1.0+0j])
        >>> distance = metrics.trace_distance(state1, state2)
        >>> print(distance)  # Should be 1.0 (orthogonal states)
        """
        state1 = np.asarray(state1, dtype=complex)
        state2 = np.asarray(state2, dtype=complex)

        # Compute fidelity
        fidelity = np.abs(np.vdot(state1, state2)) ** 2

        # Trace distance for pure states
        distance = np.sqrt(1.0 - fidelity)

        return float(distance)

    def compare_ot_results(
        self, classical_result: Dict[str, Any], quantum_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare classical and quantum OT results.

        Parameters
        ----------
        classical_result : Dict[str, Any]
            Classical OT result dictionary
        quantum_result : Dict[str, Any]
            Quantum OT result dictionary

        Returns
        -------
        Dict[str, float]
            Comparison metrics containing:
            - 'cost_difference': Absolute cost difference
            - 'cost_ratio': Quantum cost / classical cost
            - 'relative_error': Relative error percentage

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> classical = {'cost': 1.5}
        >>> quantum = {'cost': 1.8}
        >>> comparison = metrics.compare_ot_results(classical, quantum)
        >>> print(comparison['relative_error'])
        """
        classical_cost = classical_result.get("cost", None)
        quantum_cost = quantum_result.get("cost", None)

        if classical_cost is None or quantum_cost is None:
            raise ValueError("Both results must contain 'cost' key")

        # Compute comparison metrics
        cost_difference = abs(quantum_cost - classical_cost)
        cost_ratio = quantum_cost / classical_cost if classical_cost > 0 else np.inf
        relative_error = (
            cost_difference / classical_cost * 100 if classical_cost > 0 else np.inf
        )

        comparison = {
            "classical_cost": float(classical_cost),
            "quantum_cost": float(quantum_cost),
            "cost_difference": float(cost_difference),
            "cost_ratio": float(cost_ratio),
            "relative_error": float(relative_error),
        }

        logger.debug(
            f"OT Comparison: Classical={classical_cost:.4f}, "
            f"Quantum={quantum_cost:.4f}, "
            f"Relative Error={relative_error:.2f}%"
        )

        return comparison

    def compute_marginal_error(
        self, probabilities: np.ndarray, target_marginal: np.ndarray
    ) -> float:
        """
        Compute marginal constraint violation error.

        In optimal transport, the marginal constraints require that
        the row/column sums of the transport plan match the source
        and target distributions. This function measures violation.

        Parameters
        ----------
        probabilities : np.ndarray
            Measured probability distribution
        target_marginal : np.ndarray
            Target marginal distribution

        Returns
        -------
        float
            L2 error between probabilities and target

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> probs = np.array([0.3, 0.7])
        >>> target = np.array([0.4, 0.6])
        >>> error = metrics.compute_marginal_error(probs, target)
        """
        probabilities = np.asarray(probabilities, dtype=float)
        target_marginal = np.asarray(target_marginal, dtype=float)

        if probabilities.ndim != 1 or target_marginal.ndim != 1:
            raise ValueError("Inputs must be 1D arrays")

        # Truncate or pad to match sizes
        min_len = min(len(probabilities), len(target_marginal))
        probs = probabilities[:min_len]
        target = target_marginal[:min_len]

        # L2 error
        error = np.sqrt(np.sum((probs - target) ** 2))

        return float(error)

    def compute_convergence_rate(
        self, convergence_history: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze convergence properties of optimization.

        Parameters
        ----------
        convergence_history : np.ndarray
            Array of cost values during optimization

        Returns
        -------
        Dict[str, float]
            Convergence metrics:
            - 'initial_cost': First cost value
            - 'final_cost': Last cost value
            - 'improvement': Total improvement
            - 'improvement_rate': Average improvement per iteration
            - 'converged': Whether optimization converged (bool as float)

        Examples
        --------
        >>> metrics = QuantumMetrics()
        >>> history = np.array([10.0, 8.0, 6.5, 6.0, 5.9, 5.9])
        >>> analysis = metrics.compute_convergence_rate(history)
        """
        history = np.asarray(convergence_history, dtype=float)

        if len(history) == 0:
            raise ValueError("Convergence history cannot be empty")

        initial_cost = history[0]
        final_cost = history[-1]
        improvement = initial_cost - final_cost
        improvement_rate = improvement / len(history) if len(history) > 1 else 0.0

        # Check convergence (last 5 values have small variance)
        if len(history) >= 5:
            last_values = history[-5:]
            variance = np.var(last_values)
            converged = variance < 0.01  # Threshold for convergence
        else:
            converged = False

        return {
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "improvement": float(improvement),
            "improvement_rate": float(improvement_rate),
            "converged": float(converged),
        }

    def _compute_emd_1d(
        self, probs1: np.ndarray, probs2: np.ndarray, cost_matrix: np.ndarray
    ) -> float:
        """
        Compute 1D Earth Mover's Distance (simplified).

        Parameters
        ----------
        probs1 : np.ndarray
            First probability distribution
        probs2 : np.ndarray
            Second probability distribution
        cost_matrix : np.ndarray
            Cost matrix

        Returns
        -------
        float
            EMD distance
        """
        # Simplified EMD calculation
        # For 1D with sequential costs, this is the cumulative difference
        if np.allclose(
            cost_matrix,
            np.abs(np.arange(len(probs1)).reshape(-1, 1) - np.arange(len(probs2))),
        ):
            # Sequential cost case - use efficient cumulative method
            cumsum1 = np.cumsum(probs1)
            cumsum2 = np.cumsum(probs2)
            emd = np.sum(np.abs(cumsum1 - cumsum2))
        else:
            # General case - use cost matrix
            emd = 0.0
            for i in range(len(probs1)):
                for j in range(len(probs2)):
                    emd += cost_matrix[i, j] * abs(probs1[i] - probs2[j])

        return float(emd)


def quantum_wasserstein_distance(
    state1: np.ndarray, state2: np.ndarray, cost_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Compute quantum Wasserstein distance between quantum states.

    This is a convenience function that wraps QuantumMetrics.

    Parameters
    ----------
    state1 : np.ndarray
        First quantum state
    state2 : np.ndarray
        Second quantum state
    cost_matrix : Optional[np.ndarray], optional
        Cost matrix, by default None

    Returns
    -------
    float
        Quantum Wasserstein distance

    Examples
    --------
    >>> state1 = np.array([0.7+0j, 0.3+0j, 0, 0])
    >>> state2 = np.array([0.5+0j, 0.5+0j, 0, 0])
    >>> distance = quantum_wasserstein_distance(state1, state2)
    """
    metrics = QuantumMetrics()
    return metrics.quantum_wasserstein_distance(state1, state2, cost_matrix)


def fidelity_distance(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute fidelity-based distance between quantum states.

    This is a convenience function that wraps QuantumMetrics.

    Parameters
    ----------
    state1 : np.ndarray
        First quantum state
    state2 : np.ndarray
        Second quantum state

    Returns
    -------
    float
        Fidelity distance

    Examples
    --------
    >>> state1 = np.array([1.0+0j, 0])
    >>> state2 = np.array([0.7+0j, 0.7+0j]) / np.sqrt(2)
    >>> distance = fidelity_distance(state1, state2)
    """
    metrics = QuantumMetrics()
    return metrics.fidelity_distance(state1, state2)


def compare_classical_quantum(
    classical_result: Dict[str, Any], quantum_result: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compare classical and quantum OT results.

    This is a convenience function that wraps QuantumMetrics.

    Parameters
    ----------
    classical_result : Dict[str, Any]
        Classical OT result
    quantum_result : Dict[str, Any]
        Quantum OT result

    Returns
    -------
    Dict[str, float]
        Comparison metrics

    Examples
    --------
    >>> classical = {'cost': 2.5}
    >>> quantum = {'cost': 2.8}
    >>> comparison = compare_classical_quantum(classical, quantum)
    >>> print(f"Relative error: {comparison['relative_error']:.1f}%")
    """
    metrics = QuantumMetrics()
    return metrics.compare_ot_results(classical_result, quantum_result)
