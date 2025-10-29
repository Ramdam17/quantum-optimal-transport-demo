"""
Quantum Optimal Transport algorithms module.

This module implements simplified quantum algorithms for computing optimal
transport between probability distributions. The algorithms are designed
for educational purposes and use variational quantum circuits.

IMPORTANT: This is a pedagogical implementation. The quantum algorithms
implemented here do not provide quantum advantage over classical methods
for the problem sizes we consider. They serve to demonstrate concepts
of quantum optimization applied to optimal transport.

Classes
-------
QuantumOT
    Main class for quantum optimal transport computation
VQEOptimalTransport
    Variational Quantum Eigensolver approach to OT
QAOAOptimalTransport
    Quantum Approximate Optimization Algorithm for OT

Examples
--------
>>> from src.quantum.qot_algorithms import QuantumOT
>>> import numpy as np
>>>
>>> # Create distributions
>>> source = np.array([0.3, 0.7])
>>> target = np.array([0.6, 0.4])
>>>
>>> # Compute quantum OT
>>> qot = QuantumOT(n_qubits=2, method='vqe')
>>> result = qot.compute(source, target)
>>> print(result['cost'])

Notes
-----
The quantum optimal transport problem seeks to find a quantum state
that minimizes a cost functional while preserving marginal constraints.
This is a variational problem solved using parameterized quantum circuits.

References
----------
.. [1] De Palma, G., Trevisan, D. (2021). "Quantum Optimal Transport"
       arXiv:2105.06922
.. [2] Chakrabarti, S., et al. (2019). "Quantum Wasserstein Generative
       Adversarial Networks" NeurIPS.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from src.quantum.circuits import QuantumCircuitBuilder
from src.quantum.simulators import QuantumSimulator
from src.quantum.state_preparation import StatePreparation

logger = logging.getLogger(__name__)


class QuantumOT:
    """
    Quantum Optimal Transport using variational quantum circuits.

    This class implements a simplified quantum approach to optimal transport
    using variational quantum algorithms. The transport plan is encoded in
    a parameterized quantum circuit, and the optimal parameters are found
    through classical optimization.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to use
    method : str, optional
        Optimization method: 'vqe' or 'qaoa'
        Default is 'vqe'
    max_iterations : int, optional
        Maximum optimization iterations
        Default is 100
    optimizer : str, optional
        Classical optimizer: 'COBYLA', 'SLSQP', 'BFGS', 'Nelder-Mead'
        Default is 'COBYLA'
    shots : int, optional
        Number of measurement shots
        Default is 1024
    seed : Optional[int], optional
        Random seed for reproducibility
        Default is None

    Attributes
    ----------
    n_qubits : int
        Number of qubits
    method : str
        Optimization method
    max_iterations : int
        Maximum iterations
    optimizer_name : str
        Classical optimizer name
    shots : int
        Measurement shots
    seed : Optional[int]
        Random seed
    simulator : QuantumSimulator
        Quantum simulator instance
    state_prep : StatePreparation
        State preparation instance

    Examples
    --------
    >>> qot = QuantumOT(n_qubits=4, method='vqe', max_iterations=50)
    >>> source = np.array([0.4, 0.6])
    >>> target = np.array([0.7, 0.3])
    >>> result = qot.compute(source, target)
    >>> print(f"Quantum OT cost: {result['cost']:.4f}")
    """

    VALID_METHODS = ["vqe", "qaoa"]
    VALID_OPTIMIZERS = ["COBYLA", "SLSQP", "BFGS", "Nelder-Mead"]

    def __init__(
        self,
        n_qubits: int,
        method: str = "vqe",
        max_iterations: int = 100,
        optimizer: str = "COBYLA",
        shots: int = 1024,
        seed: Optional[int] = None,
    ):
        """Initialize Quantum OT."""
        # Validate inputs
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. " f"Must be one of {self.VALID_METHODS}"
            )

        if optimizer not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer '{optimizer}'. "
                f"Must be one of {self.VALID_OPTIMIZERS}"
            )

        self.n_qubits = n_qubits
        self.method = method
        self.max_iterations = max_iterations
        self.optimizer_name = optimizer
        self.shots = shots
        self.seed = seed

        # Initialize components
        self.simulator = QuantumSimulator(backend="statevector", shots=shots, seed=seed)
        self.state_prep = StatePreparation()

        # Optimization tracking
        self.optimization_history = []
        self.iteration_count = 0

        logger.info(
            f"Initialized QuantumOT with n_qubits={n_qubits}, "
            f"method={method}, optimizer={optimizer}"
        )

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute quantum optimal transport between distributions.

        Parameters
        ----------
        source : np.ndarray
            Source probability distribution
        target : np.ndarray
            Target probability distribution
        cost_matrix : Optional[np.ndarray], optional
            Cost matrix between points. If None, uses Euclidean distance.

        Returns
        -------
        Dict[str, Any]
            Results dictionary containing:
            - 'cost': Optimal transport cost
            - 'optimal_params': Optimal circuit parameters
            - 'final_state': Final quantum state
            - 'iterations': Number of iterations
            - 'convergence_history': Cost values during optimization
            - 'execution_time': Total execution time in seconds
            - 'success': Whether optimization converged

        Examples
        --------
        >>> qot = QuantumOT(n_qubits=4)
        >>> source = np.array([0.3, 0.7])
        >>> target = np.array([0.6, 0.4])
        >>> result = qot.compute(source, target)
        """
        start_time = time.time()

        # Validate inputs
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("Source and target must be 1D arrays")

        if len(source) != len(target):
            raise ValueError("Source and target must have same length")

        # Normalize distributions
        source = source / np.sum(source)
        target = target / np.sum(target)

        # Create or validate cost matrix
        if cost_matrix is None:
            n = len(source)
            cost_matrix = self._default_cost_matrix(n)

        logger.info(
            f"Computing quantum OT: {len(source)} points, " f"method={self.method}"
        )

        # Reset optimization tracking
        self.optimization_history = []
        self.iteration_count = 0

        # Run optimization
        if self.method == "vqe":
            result = self._vqe_optimization(source, target, cost_matrix)
        else:  # qaoa
            result = self._qaoa_optimization(source, target, cost_matrix)

        # Add execution time
        result["execution_time"] = time.time() - start_time

        logger.info(
            f"Quantum OT completed: cost={result['cost']:.6f}, "
            f"iterations={result['iterations']}, "
            f"time={result['execution_time']:.2f}s"
        )

        return result

    def _default_cost_matrix(self, n: int) -> np.ndarray:
        """
        Create default cost matrix (squared Euclidean distance).

        Parameters
        ----------
        n : int
            Number of points

        Returns
        -------
        np.ndarray
            Cost matrix of shape (n, n)
        """
        # Simple 1D grid cost
        indices = np.arange(n).reshape(-1, 1)
        cost_matrix = (indices - indices.T) ** 2
        return cost_matrix.astype(float)

    def _vqe_optimization(
        self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Variational Quantum Eigensolver optimization for OT.

        Parameters
        ----------
        source : np.ndarray
            Source distribution
        target : np.ndarray
            Target distribution
        cost_matrix : np.ndarray
            Cost matrix

        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        # Number of parameters (layers × gates)
        n_layers = 3
        n_params = n_layers * self.n_qubits * 2  # RY and RZ per qubit per layer

        # Initial parameters
        if self.seed is not None:
            np.random.seed(self.seed)
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        # Define cost function
        def cost_function(params):
            """Evaluate cost for given parameters."""
            self.iteration_count += 1

            # Build parameterized circuit
            circuit = self._build_vqe_circuit(params, n_layers)

            # Simulate
            result = self.simulator.run(circuit.circuit)

            # Compute cost
            cost = self._compute_ot_cost(
                result.probabilities, source, target, cost_matrix
            )

            # Track history
            self.optimization_history.append(cost)

            if self.iteration_count % 10 == 0:
                logger.debug(f"Iteration {self.iteration_count}: cost={cost:.6f}")

            return cost

        # Run optimization
        result = minimize(
            cost_function,
            initial_params,
            method=self.optimizer_name,
            options={"maxiter": self.max_iterations},
        )

        # Get final state
        final_circuit = self._build_vqe_circuit(result.x, n_layers)
        final_result = self.simulator.run(final_circuit.circuit)

        return {
            "cost": result.fun,
            "optimal_params": result.x,
            "final_state": final_result.statevector,
            "final_probabilities": final_result.probabilities,
            "iterations": self.iteration_count,
            "convergence_history": self.optimization_history,
            "success": result.success,
            "message": result.message,
        }

    def _qaoa_optimization(
        self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        QAOA optimization for OT.

        Parameters
        ----------
        source : np.ndarray
            Source distribution
        target : np.ndarray
            Target distribution
        cost_matrix : np.ndarray
            Cost matrix

        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        # QAOA parameters: p layers of (cost Hamiltonian, mixer)
        p = 2  # QAOA depth
        n_params = 2 * p  # gamma and beta for each layer

        # Initial parameters
        if self.seed is not None:
            np.random.seed(self.seed)
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        # Define cost function
        def cost_function(params):
            """Evaluate QAOA cost."""
            self.iteration_count += 1

            # Build QAOA circuit
            circuit = self._build_qaoa_circuit(params, p, cost_matrix)

            # Simulate
            result = self.simulator.run(circuit.circuit)

            # Compute cost
            cost = self._compute_ot_cost(
                result.probabilities, source, target, cost_matrix
            )

            # Track history
            self.optimization_history.append(cost)

            if self.iteration_count % 10 == 0:
                logger.debug(f"Iteration {self.iteration_count}: cost={cost:.6f}")

            return cost

        # Run optimization
        result = minimize(
            cost_function,
            initial_params,
            method=self.optimizer_name,
            options={"maxiter": self.max_iterations},
        )

        # Get final state
        final_circuit = self._build_qaoa_circuit(result.x, p, cost_matrix)
        final_result = self.simulator.run(final_circuit.circuit)

        return {
            "cost": result.fun,
            "optimal_params": result.x,
            "final_state": final_result.statevector,
            "final_probabilities": final_result.probabilities,
            "iterations": self.iteration_count,
            "convergence_history": self.optimization_history,
            "success": result.success,
            "message": result.message,
        }

    def _build_vqe_circuit(
        self, params: np.ndarray, n_layers: int
    ) -> QuantumCircuitBuilder:
        """
        Build VQE ansatz circuit.

        Parameters
        ----------
        params : np.ndarray
            Circuit parameters
        n_layers : int
            Number of layers

        Returns
        -------
        QuantumCircuitBuilder
            Parameterized circuit
        """
        circuit = QuantumCircuitBuilder(self.n_qubits)

        # Initial superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # Variational layers
        param_idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                circuit.ry(params[param_idx], i)
                param_idx += 1
                circuit.rz(params[param_idx], i)
                param_idx += 1

            # Entangling layer (if not last layer)
            if layer < n_layers - 1:
                for i in range(self.n_qubits - 1):
                    circuit.cnot(i, i + 1)

        return circuit

    def _build_qaoa_circuit(
        self, params: np.ndarray, p: int, cost_matrix: np.ndarray
    ) -> QuantumCircuitBuilder:
        """
        Build QAOA circuit.

        Parameters
        ----------
        params : np.ndarray
            QAOA parameters [gamma_1, beta_1, ..., gamma_p, beta_p]
        p : int
            QAOA depth
        cost_matrix : np.ndarray
            Cost matrix

        Returns
        -------
        QuantumCircuitBuilder
            QAOA circuit
        """
        circuit = QuantumCircuitBuilder(self.n_qubits)

        # Initial superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # QAOA layers
        for layer in range(p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]

            # Cost Hamiltonian (simplified)
            for i in range(self.n_qubits):
                circuit.rz(gamma * 0.5, i)  # Simplified cost encoding

            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)
                circuit.rz(gamma * 0.5, i + 1)
                circuit.cnot(i, i + 1)

            # Mixer Hamiltonian
            for i in range(self.n_qubits):
                circuit.rx(beta, i)

        return circuit

    def _compute_ot_cost(
        self,
        probabilities: Optional[Dict[str, float]],
        source: np.ndarray,
        target: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> float:
        """
        Compute OT cost from quantum state probabilities.

        Parameters
        ----------
        probabilities : Optional[Dict[str, float]]
            Measurement probabilities
        source : np.ndarray
            Source distribution
        target : np.ndarray
            Target distribution
        cost_matrix : np.ndarray
            Cost matrix

        Returns
        -------
        float
            OT cost
        """
        if probabilities is None:
            return np.inf

        n = len(source)

        # Extract relevant probabilities (first n states)
        probs_array = np.zeros(min(n, 2**self.n_qubits))
        for i in range(len(probs_array)):
            bitstring = format(i, f"0{self.n_qubits}b")
            probs_array[i] = probabilities.get(bitstring, 0.0)

        # Pad or truncate to match distribution size
        if len(probs_array) > n:
            probs_array = probs_array[:n]
        elif len(probs_array) < n:
            probs_array = np.pad(probs_array, (0, n - len(probs_array)))

        # Normalize
        if np.sum(probs_array) > 0:
            probs_array = probs_array / np.sum(probs_array)
        else:
            probs_array = np.ones(n) / n

        # Compute Wasserstein-like cost
        # Cost = sum over i,j: cost_matrix[i,j] * |prob[i] - target[j]|
        cost = 0.0
        for i in range(n):
            for j in range(n):
                cost += cost_matrix[i, j] * abs(probs_array[i] - target[j])

        # Add marginal constraint penalty
        marginal_penalty = np.sum((probs_array - source) ** 2)
        cost += 10.0 * marginal_penalty  # Weight penalty

        return cost

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantumOT(n_qubits={self.n_qubits}, method='{self.method}', "
            f"optimizer='{self.optimizer_name}')"
        )
