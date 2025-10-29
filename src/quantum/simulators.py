"""
Quantum circuit simulation module.

This module provides wrappers around Qiskit Aer simulators for executing
quantum circuits. It supports both statevector simulation (exact amplitudes)
and shot-based simulation (sampling from measurement outcomes).

Classes
-------
QuantumSimulator
    Main simulator class for executing quantum circuits
SimulationResult
    Container for simulation results with convenient access methods

Examples
--------
>>> from qiskit import QuantumCircuit
>>> from src.quantum.simulators import QuantumSimulator
>>>
>>> # Create a simple circuit
>>> circuit = QuantumCircuit(2, 2)
>>> circuit.h(0)
>>> circuit.cx(0, 1)
>>> circuit.measure([0, 1], [0, 1])
>>>
>>> # Run with statevector simulator
>>> sim = QuantumSimulator(backend='statevector')
>>> result = sim.run(circuit)
>>> print(result.statevector)
>>>
>>> # Run with shot-based simulator
>>> sim = QuantumSimulator(backend='qasm', shots=1024)
>>> result = sim.run(circuit)
>>> print(result.counts)

Notes
-----
This module wraps Qiskit Aer simulators to provide a consistent interface
for quantum circuit execution. The statevector backend is exact but limited
to ~30 qubits. The qasm backend supports more qubits but requires shots.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator

logger = logging.getLogger(__name__)


class SimulationResult:
    """
    Container for quantum simulation results.

    This class wraps the results from quantum circuit execution, providing
    convenient access to statevectors, measurement counts, and other data.

    Parameters
    ----------
    raw_result : qiskit.result.Result
        Raw result object from Qiskit execution
    backend_name : str
        Name of the backend used for simulation
    shots : Optional[int]
        Number of shots used (None for statevector simulation)

    Attributes
    ----------
    raw_result : qiskit.result.Result
        Original Qiskit result object
    backend_name : str
        Backend used for execution
    shots : Optional[int]
        Number of measurement shots
    counts : Optional[Dict[str, int]]
        Measurement outcome counts (qasm backend only)
    statevector : Optional[np.ndarray]
        State vector amplitudes (statevector backend only)
    probabilities : Optional[Dict[str, float]]
        Measurement outcome probabilities

    Examples
    --------
    >>> result = sim.run(circuit)
    >>> print(f"Backend: {result.backend_name}")
    >>> print(f"Counts: {result.counts}")
    >>> print(f"Probabilities: {result.probabilities}")
    """

    def __init__(self, raw_result, backend_name: str, shots: Optional[int] = None):
        """Initialize simulation result."""
        self.raw_result = raw_result
        self.backend_name = backend_name
        self.shots = shots

        # Extract counts if available
        self._counts = None
        self._statevector = None
        self._probabilities = None

        # Process results based on backend type
        if "statevector" in backend_name.lower():
            self._extract_statevector()
        elif "qasm" in backend_name.lower():
            self._extract_counts()

    def _extract_statevector(self):
        """Extract statevector from result."""
        try:
            # Get statevector from result
            if hasattr(self.raw_result, "_statevector_data"):
                # Our custom statevector result
                self._statevector = np.array(self.raw_result._statevector_data)
            elif (
                hasattr(self.raw_result, "results") and len(self.raw_result.results) > 0
            ):
                result_data = self.raw_result.results[0]
                if hasattr(result_data, "data") and "statevector" in result_data.data:
                    self._statevector = np.array(result_data.data["statevector"])
                elif hasattr(self.raw_result, "get_statevector"):
                    self._statevector = np.array(self.raw_result.get_statevector())

            if self._statevector is None:
                logger.debug("No statevector found in result")
                return

            # Compute probabilities from statevector
            probs = np.abs(self._statevector) ** 2
            n_qubits = int(np.log2(len(probs)))
            self._probabilities = {
                format(i, f"0{n_qubits}b"): float(probs[i])
                for i in range(len(probs))
                if probs[i] > 1e-10  # Filter negligible probabilities
            }
        except Exception as e:
            logger.warning(f"Could not extract statevector: {e}")

    def _extract_counts(self):
        """Extract measurement counts from result."""
        try:
            self._counts = self.raw_result.get_counts()
            # Compute probabilities from counts
            total = sum(self._counts.values())
            self._probabilities = {
                key: count / total for key, count in self._counts.items()
            }
        except Exception as e:
            logger.warning(f"Could not extract counts: {e}")

    @property
    def counts(self) -> Optional[Dict[str, int]]:
        """Get measurement counts."""
        return self._counts

    @property
    def statevector(self) -> Optional[np.ndarray]:
        """Get statevector amplitudes."""
        return self._statevector

    @property
    def probabilities(self) -> Optional[Dict[str, float]]:
        """Get measurement probabilities."""
        return self._probabilities

    def get_probability(self, bitstring: str) -> float:
        """
        Get probability of a specific measurement outcome.

        Parameters
        ----------
        bitstring : str
            Measurement outcome (e.g., '00', '01', '10', '11')

        Returns
        -------
        float
            Probability of the outcome
        """
        if self._probabilities is None:
            return 0.0
        return self._probabilities.get(bitstring, 0.0)

    def get_expectation_value(self, observable: np.ndarray) -> complex:
        """
        Compute expectation value of an observable.

        Parameters
        ----------
        observable : np.ndarray
            Observable operator as a matrix

        Returns
        -------
        complex
            Expectation value <ψ|O|ψ>

        Raises
        ------
        ValueError
            If statevector is not available
        """
        if self._statevector is None:
            raise ValueError("Expectation value requires statevector simulation")

        psi = self._statevector
        expectation = np.conj(psi) @ observable @ psi
        return expectation

    def __repr__(self) -> str:
        """String representation."""
        info = [f"SimulationResult(backend='{self.backend_name}'"]
        if self.shots:
            info.append(f"shots={self.shots}")
        if self._counts:
            info.append(f"n_outcomes={len(self._counts)}")
        if self._statevector is not None:
            info.append(f"n_amplitudes={len(self._statevector)}")
        return ", ".join(info) + ")"


class QuantumSimulator:
    """
    Quantum circuit simulator with multiple backend support.

    This class provides a unified interface for executing quantum circuits
    using different Qiskit Aer backends. It supports statevector simulation
    (exact) and qasm simulation (shot-based sampling).

    Parameters
    ----------
    backend : str, optional
        Backend name: 'statevector', 'qasm', or 'aer_simulator'
        Default is 'statevector'
    shots : int, optional
        Number of measurement shots for qasm backend
        Default is 1024
    optimization_level : int, optional
        Transpilation optimization level (0-3)
        Default is 1
    seed : Optional[int], optional
        Random seed for reproducibility
        Default is None

    Attributes
    ----------
    backend_name : str
        Name of the selected backend
    backend : qiskit_aer.AerSimulator
        Backend instance
    shots : int
        Number of shots for sampling
    optimization_level : int
        Transpilation optimization level
    seed : Optional[int]
        Random seed

    Examples
    --------
    >>> # Statevector simulation (exact)
    >>> sim = QuantumSimulator(backend='statevector')
    >>> result = sim.run(circuit)
    >>> print(result.statevector)
    >>>
    >>> # Shot-based simulation
    >>> sim = QuantumSimulator(backend='qasm', shots=2048, seed=42)
    >>> result = sim.run(circuit)
    >>> print(result.counts)
    >>>
    >>> # Run multiple circuits
    >>> results = sim.run_batch([circuit1, circuit2, circuit3])
    """

    VALID_BACKENDS = ["statevector", "qasm", "aer_simulator"]

    def __init__(
        self,
        backend: str = "statevector",
        shots: int = 1024,
        optimization_level: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize quantum simulator."""
        # Validate backend
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. " f"Must be one of {self.VALID_BACKENDS}"
            )

        self.backend_name = backend
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed = seed

        # Initialize backend
        self.backend = self._initialize_backend()

        logger.info(
            f"Initialized QuantumSimulator with backend='{backend}', "
            f"shots={shots}, seed={seed}"
        )

    def _initialize_backend(self) -> AerSimulator:
        """
        Initialize the Qiskit Aer backend.

        Returns
        -------
        AerSimulator
            Initialized backend instance
        """
        if self.backend_name == "statevector":
            # Save statevector by default
            backend = AerSimulator(method="statevector")
        elif self.backend_name == "qasm":
            backend = AerSimulator(method="automatic")
        else:  # aer_simulator
            backend = AerSimulator()

        return backend

    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Execute a quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute
        shots : Optional[int], optional
            Number of shots (overrides instance default)
        seed : Optional[int], optional
            Random seed (overrides instance default)

        Returns
        -------
        SimulationResult
            Simulation results

        Raises
        ------
        ValueError
            If circuit is invalid or execution fails

        Examples
        --------
        >>> circuit = QuantumCircuit(2, 2)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> circuit.measure([0, 1], [0, 1])
        >>>
        >>> sim = QuantumSimulator(backend='qasm')
        >>> result = sim.run(circuit, shots=2048)
        >>> print(result.counts)
        """
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("Input must be a QuantumCircuit")

        # Use provided values or defaults
        run_shots = shots if shots is not None else self.shots
        run_seed = seed if seed is not None else self.seed

        # Execute based on backend type
        if self.backend_name == "statevector":
            # Use Statevector class for true statevector simulation
            sv = Statevector.from_instruction(circuit)
            # Store statevector directly as a simple attribute
            raw_result = type(
                "obj",
                (object,),
                {
                    "backend_name": "statevector_simulator",
                    "success": True,
                    "_statevector_data": sv.data,
                },
            )()
        else:
            # Transpile circuit for backend
            transpiled = transpile(
                circuit,
                backend=self.backend,
                optimization_level=self.optimization_level,
                seed_transpiler=run_seed,
            )

            # Shot-based simulation
            job = self.backend.run(transpiled, shots=run_shots, seed_simulator=run_seed)
            raw_result = job.result()

        logger.debug(
            f"Executed circuit with {circuit.num_qubits} qubits, "
            f"{circuit.depth()} depth"
        )

        return SimulationResult(
            raw_result=raw_result,
            backend_name=self.backend_name,
            shots=run_shots if self.backend_name != "statevector" else None,
        )

    def run_batch(
        self,
        circuits: List[QuantumCircuit],
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Execute multiple quantum circuits.

        Parameters
        ----------
        circuits : List[QuantumCircuit]
            List of circuits to execute
        shots : Optional[int], optional
            Number of shots per circuit
        seed : Optional[int], optional
            Random seed

        Returns
        -------
        List[SimulationResult]
            List of simulation results

        Examples
        --------
        >>> circuits = [circuit1, circuit2, circuit3]
        >>> sim = QuantumSimulator(backend='statevector')
        >>> results = sim.run_batch(circuits)
        >>> for i, result in enumerate(results):
        ...     print(f"Circuit {i}: {result.probabilities}")
        """
        if not circuits:
            return []

        logger.info(f"Executing batch of {len(circuits)} circuits")

        results = []
        for i, circuit in enumerate(circuits):
            try:
                result = self.run(circuit, shots=shots, seed=seed)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute circuit {i}: {e}")
                raise

        return results

    def get_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Get statevector from circuit execution.

        This is a convenience method that automatically uses statevector
        backend regardless of the simulator's configuration.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute

        Returns
        -------
        np.ndarray
            Statevector as complex array

        Raises
        ------
        ValueError
            If circuit has measurements or statevector extraction fails

        Examples
        --------
        >>> circuit = QuantumCircuit(2)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>>
        >>> sim = QuantumSimulator()
        >>> sv = sim.get_statevector(circuit)
        >>> print(np.abs(sv)**2)  # Probabilities
        """
        # Check if circuit has measurements
        if circuit.num_clbits > 0:
            logger.warning(
                "Circuit has classical bits. Creating copy without measurements."
            )
            circuit_copy = circuit.remove_final_measurements(inplace=False)
            if circuit_copy is None:
                circuit_copy = circuit.copy()
        else:
            circuit_copy = circuit

        # Use Statevector class directly
        sv = Statevector.from_instruction(circuit_copy)
        return np.array(sv.data)

    def get_counts(
        self, circuit: QuantumCircuit, shots: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get measurement counts from circuit execution.

        This is a convenience method that automatically uses qasm backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute (must have measurements)
        shots : Optional[int], optional
            Number of shots

        Returns
        -------
        Dict[str, int]
            Measurement counts

        Raises
        ------
        ValueError
            If circuit has no measurements

        Examples
        --------
        >>> circuit = QuantumCircuit(2, 2)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> circuit.measure([0, 1], [0, 1])
        >>>
        >>> sim = QuantumSimulator()
        >>> counts = sim.get_counts(circuit, shots=2048)
        >>> print(counts)
        """
        if circuit.num_clbits == 0:
            raise ValueError("Circuit must have measurements to get counts")

        run_shots = shots if shots is not None else self.shots

        # Use qasm backend
        temp_backend = AerSimulator(method="automatic")
        transpiled = transpile(
            circuit, backend=temp_backend, optimization_level=self.optimization_level
        )

        job = temp_backend.run(transpiled, shots=run_shots, seed_simulator=self.seed)
        result = job.result()

        return result.get_counts()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantumSimulator(backend='{self.backend_name}', "
            f"shots={self.shots}, seed={self.seed})"
        )
