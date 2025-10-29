"""
Quantum state preparation module.

This module provides methods for encoding classical probability distributions
into quantum states using various encoding schemes. It is essential for
quantum optimal transport algorithms that require probability distributions
to be represented as quantum amplitudes.

Classes
-------
StatePreparation
    Main class for preparing quantum states from classical data

Functions
---------
amplitude_encoding
    Encode probability distribution as quantum amplitudes
basis_encoding
    Encode data using computational basis states
angle_encoding
    Encode data as rotation angles

Examples
--------
>>> import numpy as np
>>> from src.quantum.state_preparation import StatePreparation
>>>
>>> # Create a probability distribution
>>> probs = np.array([0.25, 0.25, 0.25, 0.25])
>>>
>>> # Prepare quantum state
>>> prep = StatePreparation()
>>> circuit = prep.prepare_state(probs, method='amplitude')
>>> print(circuit.circuit)

Notes
-----
Amplitude encoding is the most natural representation for probability
distributions in quantum computing, as the squared amplitudes directly
correspond to measurement probabilities. However, it requires state
preparation circuits that can be complex for arbitrary distributions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector

from src.quantum.circuits import QuantumCircuitBuilder

logger = logging.getLogger(__name__)


class StatePreparation:
    """
    Quantum state preparation from classical probability distributions.

    This class provides methods to encode classical probability distributions
    into quantum states using different encoding schemes. The primary method
    is amplitude encoding, where probabilities are encoded as squared amplitudes.

    Parameters
    ----------
    normalization : str, optional
        Normalization method: 'l1' (probabilities) or 'l2' (amplitudes)
        Default is 'l1'
    tolerance : float, optional
        Tolerance for validation checks
        Default is 1e-6

    Attributes
    ----------
    normalization : str
        Normalization method
    tolerance : float
        Validation tolerance

    Examples
    --------
    >>> prep = StatePreparation()
    >>> probs = np.array([0.5, 0.3, 0.2])
    >>> circuit = prep.prepare_state(probs, method='amplitude')
    >>>
    >>> # Verify the state
    >>> from src.quantum.simulators import QuantumSimulator
    >>> sim = QuantumSimulator()
    >>> result = sim.run(circuit.circuit)
    >>> print(result.probabilities)
    """

    VALID_METHODS = ["amplitude", "basis", "angle"]
    VALID_NORMALIZATIONS = ["l1", "l2"]

    def __init__(self, normalization: str = "l1", tolerance: float = 1e-6):
        """Initialize state preparation."""
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(
                f"Invalid normalization '{normalization}'. "
                f"Must be one of {self.VALID_NORMALIZATIONS}"
            )

        self.normalization = normalization
        self.tolerance = tolerance

        logger.info(
            f"Initialized StatePreparation with "
            f"normalization='{normalization}', tolerance={tolerance}"
        )

    def prepare_state(
        self, distribution: np.ndarray, method: str = "amplitude", **kwargs
    ) -> QuantumCircuitBuilder:
        """
        Prepare quantum state from probability distribution.

        Parameters
        ----------
        distribution : np.ndarray
            Probability distribution or data to encode
        method : str, optional
            Encoding method: 'amplitude', 'basis', or 'angle'
            Default is 'amplitude'
        **kwargs
            Additional method-specific parameters

        Returns
        -------
        QuantumCircuitBuilder
            Quantum circuit that prepares the state

        Raises
        ------
        ValueError
            If distribution is invalid or method is unknown

        Examples
        --------
        >>> prep = StatePreparation()
        >>> probs = np.array([0.6, 0.4])
        >>> circuit = prep.prepare_state(probs, method='amplitude')
        """
        # Validate method
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. " f"Must be one of {self.VALID_METHODS}"
            )

        # Validate distribution
        distribution = np.asarray(distribution, dtype=float)
        if distribution.ndim != 1:
            raise ValueError(f"Distribution must be 1D, got shape {distribution.shape}")

        if np.any(distribution < 0):
            raise ValueError("Distribution must be non-negative")

        if np.all(distribution == 0):
            raise ValueError("Distribution cannot be all zeros")

        # Route to appropriate method
        if method == "amplitude":
            return self._amplitude_encoding(distribution, **kwargs)
        elif method == "basis":
            return self._basis_encoding(distribution, **kwargs)
        elif method == "angle":
            return self._angle_encoding(distribution, **kwargs)

    def _amplitude_encoding(
        self, distribution: np.ndarray, pad_to_power_of_2: bool = True
    ) -> QuantumCircuitBuilder:
        """
        Encode distribution using amplitude encoding.

        In amplitude encoding, the probability distribution is encoded as the
        squared amplitudes of a quantum state: |ψ⟩ = Σᵢ √pᵢ |i⟩

        Parameters
        ----------
        distribution : np.ndarray
            Probability distribution
        pad_to_power_of_2 : bool, optional
            Whether to pad distribution to power of 2
            Default is True

        Returns
        -------
        QuantumCircuitBuilder
            Circuit that prepares the amplitude-encoded state
        """
        # Normalize distribution
        if self.normalization == "l1":
            # Probabilities: sum to 1
            distribution = distribution / np.sum(distribution)
            # Convert to amplitudes
            amplitudes = np.sqrt(distribution)
        else:  # l2
            # Amplitudes: norm = 1
            amplitudes = distribution / np.linalg.norm(distribution)

        # Pad to power of 2 if needed
        n = len(amplitudes)
        if pad_to_power_of_2:
            n_qubits = int(np.ceil(np.log2(n)))
            n_padded = 2**n_qubits
            if n_padded > n:
                amplitudes = np.pad(amplitudes, (0, n_padded - n), mode="constant")
                # Renormalize after padding
                amplitudes = amplitudes / np.linalg.norm(amplitudes)
        else:
            n_qubits = int(np.ceil(np.log2(n)))

        logger.debug(f"Amplitude encoding: {n} values -> {n_qubits} qubits")

        # Create circuit using Initialize instruction
        circuit = QuantumCircuit(n_qubits)
        initialize_gate = Initialize(amplitudes)
        circuit.append(initialize_gate, range(n_qubits))

        # Wrap in QuantumCircuitBuilder
        builder = QuantumCircuitBuilder(n_qubits)
        builder.circuit = circuit

        return builder

    def _basis_encoding(
        self, distribution: np.ndarray, n_qubits: Optional[int] = None
    ) -> QuantumCircuitBuilder:
        """
        Encode distribution using basis encoding.

        In basis encoding, each classical value is encoded as a computational
        basis state. This is more suitable for discrete data than probabilities.

        Parameters
        ----------
        distribution : np.ndarray
            Data values to encode
        n_qubits : Optional[int], optional
            Number of qubits (inferred if None)

        Returns
        -------
        QuantumCircuitBuilder
            Circuit that prepares the basis-encoded state
        """
        # Determine number of qubits needed
        max_value = int(np.max(distribution))
        if n_qubits is None:
            n_qubits = int(np.ceil(np.log2(max_value + 1)))

        logger.debug(f"Basis encoding: max value {max_value} -> {n_qubits} qubits")

        # Create uniform superposition of encoded states
        # For simplicity, create equal superposition
        n_states = 2**n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)

        for i, val in enumerate(distribution[:n_states]):
            idx = int(val) % n_states
            amplitudes[idx] += 1.0

        # Normalize
        if np.linalg.norm(amplitudes) > 0:
            amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Create circuit
        circuit = QuantumCircuit(n_qubits)
        initialize_gate = Initialize(amplitudes)
        circuit.append(initialize_gate, range(n_qubits))

        builder = QuantumCircuitBuilder(n_qubits)
        builder.circuit = circuit

        return builder

    def _angle_encoding(
        self, distribution: np.ndarray, angle_range: Tuple[float, float] = (0, np.pi)
    ) -> QuantumCircuitBuilder:
        """
        Encode distribution using angle encoding.

        In angle encoding, classical values are encoded as rotation angles
        applied to qubits. Each value controls a rotation gate.

        Parameters
        ----------
        distribution : np.ndarray
            Data values to encode
        angle_range : Tuple[float, float], optional
            Range for angle mapping (min, max)
            Default is (0, π)

        Returns
        -------
        QuantumCircuitBuilder
            Circuit that prepares the angle-encoded state
        """
        # Normalize distribution to angle range
        min_val, max_val = np.min(distribution), np.max(distribution)
        if max_val > min_val:
            normalized = (distribution - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(distribution) * 0.5

        angles = angle_range[0] + normalized * (angle_range[1] - angle_range[0])

        # Each value gets its own qubit
        n_qubits = len(angles)

        logger.debug(f"Angle encoding: {len(angles)} values -> {n_qubits} qubits")

        # Create circuit with RY rotations
        builder = QuantumCircuitBuilder(n_qubits)
        for i, angle in enumerate(angles):
            builder.ry(angle, i)

        return builder

    def validate_state(
        self,
        circuit: Union[QuantumCircuit, QuantumCircuitBuilder],
        expected_distribution: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Validate prepared quantum state.

        Parameters
        ----------
        circuit : Union[QuantumCircuit, QuantumCircuitBuilder]
            Circuit to validate
        expected_distribution : Optional[np.ndarray], optional
            Expected probability distribution

        Returns
        -------
        Dict[str, Union[bool, float, np.ndarray]]
            Validation results with keys:
            - 'valid': bool
            - 'fidelity': float (if expected_distribution provided)
            - 'probabilities': np.ndarray
            - 'norm': float

        Examples
        --------
        >>> prep = StatePreparation()
        >>> probs = np.array([0.5, 0.5])
        >>> circuit = prep.prepare_state(probs)
        >>> validation = prep.validate_state(circuit, probs)
        >>> print(validation['valid'])
        True
        """
        # Extract circuit
        if isinstance(circuit, QuantumCircuitBuilder):
            qc = circuit.circuit
        else:
            qc = circuit

        # Get statevector
        try:
            sv = Statevector.from_instruction(qc)
            amplitudes = sv.data
        except Exception as e:
            logger.error(f"Failed to get statevector: {e}")
            return {"valid": False, "error": str(e)}

        # Compute probabilities
        probabilities = np.abs(amplitudes) ** 2

        # Check normalization
        norm = np.linalg.norm(amplitudes)
        is_normalized = np.abs(norm - 1.0) < self.tolerance

        result = {"valid": is_normalized, "probabilities": probabilities, "norm": norm}

        # Check fidelity if expected distribution provided
        if expected_distribution is not None:
            expected = np.asarray(expected_distribution)
            # Normalize
            expected = expected / np.sum(expected)
            # Pad if needed
            if len(expected) < len(probabilities):
                expected = np.pad(expected, (0, len(probabilities) - len(expected)))
            elif len(expected) > len(probabilities):
                expected = expected[: len(probabilities)]

            # Compute fidelity (overlap of probabilities)
            fidelity = np.sum(np.sqrt(probabilities * expected))
            result["fidelity"] = fidelity
            result["valid"] = result["valid"] and (fidelity > 1 - self.tolerance)

        return result

    def prepare_batch(
        self, distributions: List[np.ndarray], method: str = "amplitude", **kwargs
    ) -> List[QuantumCircuitBuilder]:
        """
        Prepare multiple quantum states from distributions.

        Parameters
        ----------
        distributions : List[np.ndarray]
            List of probability distributions
        method : str, optional
            Encoding method
        **kwargs
            Additional parameters

        Returns
        -------
        List[QuantumCircuitBuilder]
            List of prepared circuits

        Examples
        --------
        >>> prep = StatePreparation()
        >>> dists = [np.array([0.7, 0.3]), np.array([0.4, 0.6])]
        >>> circuits = prep.prepare_batch(dists)
        >>> print(len(circuits))
        2
        """
        logger.info(f"Preparing batch of {len(distributions)} states")

        circuits = []
        for i, dist in enumerate(distributions):
            try:
                circuit = self.prepare_state(dist, method=method, **kwargs)
                circuits.append(circuit)
            except Exception as e:
                logger.error(f"Failed to prepare state {i}: {e}")
                raise

        return circuits

    def compute_required_qubits(self, n_values: int, method: str = "amplitude") -> int:
        """
        Compute number of qubits required for encoding.

        Parameters
        ----------
        n_values : int
            Number of values in distribution
        method : str, optional
            Encoding method

        Returns
        -------
        int
            Number of qubits required

        Examples
        --------
        >>> prep = StatePreparation()
        >>> n_qubits = prep.compute_required_qubits(100, method='amplitude')
        >>> print(n_qubits)
        7
        """
        if method == "amplitude":
            return int(np.ceil(np.log2(n_values)))
        elif method == "basis":
            return int(np.ceil(np.log2(n_values)))
        elif method == "angle":
            return n_values  # One qubit per value
        else:
            raise ValueError(f"Unknown method: {method}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StatePreparation(normalization='{self.normalization}', "
            f"tolerance={self.tolerance})"
        )


def amplitude_encoding(
    distribution: np.ndarray, n_qubits: Optional[int] = None
) -> QuantumCircuit:
    """
    Encode probability distribution using amplitude encoding.

    Convenience function for amplitude encoding without creating a
    StatePreparation instance.

    Parameters
    ----------
    distribution : np.ndarray
        Probability distribution
    n_qubits : Optional[int], optional
        Number of qubits (inferred if None)

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the amplitude-encoded state

    Examples
    --------
    >>> probs = np.array([0.25, 0.25, 0.25, 0.25])
    >>> circuit = amplitude_encoding(probs)
    >>> print(circuit.num_qubits)
    2
    """
    prep = StatePreparation()
    builder = prep.prepare_state(distribution, method="amplitude")
    return builder.circuit


def basis_encoding(data: np.ndarray, n_qubits: Optional[int] = None) -> QuantumCircuit:
    """
    Encode data using basis encoding.

    Convenience function for basis encoding without creating a
    StatePreparation instance.

    Parameters
    ----------
    data : np.ndarray
        Data values to encode
    n_qubits : Optional[int], optional
        Number of qubits (inferred if None)

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the basis-encoded state

    Examples
    --------
    >>> data = np.array([0, 1, 2, 3])
    >>> circuit = basis_encoding(data)
    >>> print(circuit.num_qubits)
    2
    """
    prep = StatePreparation()
    builder = prep.prepare_state(data, method="basis", n_qubits=n_qubits)
    return builder.circuit


def angle_encoding(
    data: np.ndarray, angle_range: Tuple[float, float] = (0, np.pi)
) -> QuantumCircuit:
    """
    Encode data using angle encoding.

    Convenience function for angle encoding without creating a
    StatePreparation instance.

    Parameters
    ----------
    data : np.ndarray
        Data values to encode
    angle_range : Tuple[float, float], optional
        Range for angle mapping

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the angle-encoded state

    Examples
    --------
    >>> data = np.array([0.1, 0.5, 0.9])
    >>> circuit = angle_encoding(data)
    >>> print(circuit.num_qubits)
    3
    """
    prep = StatePreparation()
    builder = prep.prepare_state(data, method="angle", angle_range=angle_range)
    return builder.circuit
