"""
Quantum circuit construction and manipulation utilities.

This module provides a wrapper around Qiskit circuits with convenient
methods for building, visualizing, and analyzing quantum circuits.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class QuantumCircuitBuilder:
    """
    Builder class for quantum circuits with convenient methods.

    This class provides a high-level interface for constructing quantum
    circuits using Qiskit, with methods for common gate operations,
    parameterized circuits, and circuit analysis.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit
    n_classical : Optional[int], optional
        Number of classical bits for measurement, by default None
    name : str, optional
        Name of the circuit, by default "circuit"

    Attributes
    ----------
    circuit : QuantumCircuit
        The underlying Qiskit quantum circuit
    n_qubits : int
        Number of qubits
    n_classical : int
        Number of classical bits

    Examples
    --------
    >>> builder = QuantumCircuitBuilder(n_qubits=2)
    >>> builder.h(0)  # Hadamard on qubit 0
    >>> builder.cnot(0, 1)  # CNOT from qubit 0 to 1
    >>> print(builder.depth())
    2
    """

    def __init__(
        self, n_qubits: int, n_classical: Optional[int] = None, name: str = "circuit"
    ):
        """Initialize quantum circuit builder."""
        if n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.n_qubits = n_qubits
        self.n_classical = n_classical if n_classical is not None else n_qubits

        # Create quantum and classical registers
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(self.n_classical, "c")

        self.circuit = QuantumCircuit(qr, cr, name=name)

        logger.debug(
            f"Created quantum circuit: {n_qubits} qubits, "
            f"{self.n_classical} classical bits"
        )

    # ========================================================================
    # Single-qubit gates
    # ========================================================================

    def h(self, qubit: int) -> "QuantumCircuitBuilder":
        """
        Apply Hadamard gate to qubit.

        Parameters
        ----------
        qubit : int
            Target qubit index

        Returns
        -------
        QuantumCircuitBuilder
            Self for method chaining
        """
        self.circuit.h(qubit)
        return self

    def x(self, qubit: int) -> "QuantumCircuitBuilder":
        """Apply Pauli-X (NOT) gate."""
        self.circuit.x(qubit)
        return self

    def y(self, qubit: int) -> "QuantumCircuitBuilder":
        """Apply Pauli-Y gate."""
        self.circuit.y(qubit)
        return self

    def z(self, qubit: int) -> "QuantumCircuitBuilder":
        """Apply Pauli-Z gate."""
        self.circuit.z(qubit)
        return self

    def rx(self, angle: float, qubit: int) -> "QuantumCircuitBuilder":
        """
        Apply rotation around X-axis.

        Parameters
        ----------
        angle : float
            Rotation angle in radians
        qubit : int
            Target qubit
        """
        self.circuit.rx(angle, qubit)
        return self

    def ry(self, angle: float, qubit: int) -> "QuantumCircuitBuilder":
        """Apply rotation around Y-axis."""
        self.circuit.ry(angle, qubit)
        return self

    def rz(self, angle: float, qubit: int) -> "QuantumCircuitBuilder":
        """Apply rotation around Z-axis."""
        self.circuit.rz(angle, qubit)
        return self

    # ========================================================================
    # Two-qubit gates
    # ========================================================================

    def cnot(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """
        Apply CNOT (controlled-X) gate.

        Parameters
        ----------
        control : int
            Control qubit index
        target : int
            Target qubit index
        """
        self.circuit.cx(control, target)
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuitBuilder":
        """Apply controlled-Z gate."""
        self.circuit.cz(control, target)
        return self

    def swap(self, qubit1: int, qubit2: int) -> "QuantumCircuitBuilder":
        """Apply SWAP gate."""
        self.circuit.swap(qubit1, qubit2)
        return self

    # ========================================================================
    # Measurement
    # ========================================================================

    def measure_all(self) -> "QuantumCircuitBuilder":
        """
        Measure all qubits to classical register.

        Returns
        -------
        QuantumCircuitBuilder
            Self for method chaining
        """
        self.circuit.measure_all()
        return self

    def measure(
        self,
        qubits: Union[int, List[int]],
        classical_bits: Optional[Union[int, List[int]]] = None,
    ) -> "QuantumCircuitBuilder":
        """
        Measure specific qubits.

        Parameters
        ----------
        qubits : int or List[int]
            Qubit(s) to measure
        classical_bits : int or List[int], optional
            Classical bit(s) to store results, by default same as qubits
        """
        if isinstance(qubits, int):
            qubits = [qubits]

        if classical_bits is None:
            classical_bits = qubits
        elif isinstance(classical_bits, int):
            classical_bits = [classical_bits]

        for q, c in zip(qubits, classical_bits):
            self.circuit.measure(q, c)

        return self

    # ========================================================================
    # Circuit properties and analysis
    # ========================================================================

    def depth(self) -> int:
        """
        Get circuit depth.

        Returns
        -------
        int
            Circuit depth (number of time steps)
        """
        return self.circuit.depth()

    def gate_count(self) -> dict:
        """
        Count gates by type.

        Returns
        -------
        dict
            Dictionary mapping gate names to counts
        """
        return self.circuit.count_ops()

    def num_parameters(self) -> int:
        """
        Get number of free parameters in circuit.

        Returns
        -------
        int
            Number of unbound parameters
        """
        return len(self.circuit.parameters)

    def get_statevector(self) -> np.ndarray:
        """
        Get statevector for circuit (no measurements).

        Returns
        -------
        np.ndarray
            Complex statevector of shape (2^n_qubits,)

        Raises
        ------
        ValueError
            If circuit contains measurements
        """
        if self.circuit.num_clbits > 0 and any(
            instr.operation.name == "measure" for instr in self.circuit.data
        ):
            raise ValueError("Cannot get statevector for circuit with measurements")

        # Create statevector from circuit
        sv = Statevector.from_instruction(self.circuit)
        return sv.data

    def get_unitary(self) -> np.ndarray:
        """
        Get unitary matrix for circuit (no measurements).

        Returns
        -------
        np.ndarray
            Unitary matrix of shape (2^n_qubits, 2^n_qubits)
        """
        from qiskit.quantum_info import Operator

        if any(instr.operation.name == "measure" for instr in self.circuit.data):
            raise ValueError("Cannot get unitary for circuit with measurements")

        op = Operator(self.circuit)
        return op.data

    # ========================================================================
    # Parameterized circuits
    # ========================================================================

    def add_parameterized_layer(
        self, layer_type: str = "ry", prefix: str = "theta"
    ) -> List[Parameter]:
        """
        Add a layer of parameterized rotation gates.

        Parameters
        ----------
        layer_type : str, optional
            Type of rotation ('rx', 'ry', 'rz'), by default 'ry'
        prefix : str, optional
            Prefix for parameter names, by default 'theta'

        Returns
        -------
        List[Parameter]
            List of created parameters
        """
        params = []

        for i in range(self.n_qubits):
            param = Parameter(f"{prefix}_{i}")
            params.append(param)

            if layer_type == "rx":
                self.circuit.rx(param, i)
            elif layer_type == "ry":
                self.circuit.ry(param, i)
            elif layer_type == "rz":
                self.circuit.rz(param, i)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        logger.debug(
            f"Added parameterized {layer_type} layer with {len(params)} parameters"
        )
        return params

    def add_entangling_layer(
        self, entangling_type: str = "linear"
    ) -> "QuantumCircuitBuilder":
        """
        Add entangling layer between qubits.

        Parameters
        ----------
        entangling_type : str, optional
            Type of entanglement pattern:
            - 'linear': CNOTs in chain (0->1, 1->2, ...)
            - 'circular': Linear + wrap-around
            - 'full': All-to-all CNOTs
            By default 'linear'

        Returns
        -------
        QuantumCircuitBuilder
            Self for method chaining
        """
        if entangling_type == "linear":
            for i in range(self.n_qubits - 1):
                self.cnot(i, i + 1)

        elif entangling_type == "circular":
            for i in range(self.n_qubits):
                self.cnot(i, (i + 1) % self.n_qubits)

        elif entangling_type == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    self.cnot(i, j)

        else:
            raise ValueError(f"Unknown entangling type: {entangling_type}")

        logger.debug(f"Added {entangling_type} entangling layer")
        return self

    def bind_parameters(
        self, parameter_values: Union[List[float], np.ndarray, dict]
    ) -> QuantumCircuit:
        """
        Bind parameter values to circuit.

        Parameters
        ----------
        parameter_values : List[float] or np.ndarray or dict
            Values to bind to parameters. Can be:
            - List/array: binds in order of circuit.parameters
            - Dict: maps parameter names or objects to values

        Returns
        -------
        QuantumCircuit
            New circuit with parameters bound
        """
        if isinstance(parameter_values, dict):
            return self.circuit.assign_parameters(parameter_values)
        else:
            # Bind by position
            params = list(self.circuit.parameters)
            param_dict = dict(zip(params, parameter_values))
            return self.circuit.assign_parameters(param_dict)

    # ========================================================================
    # Visualization and representation
    # ========================================================================

    def draw(
        self, output: str = "text", filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Draw the circuit.

        Parameters
        ----------
        output : str, optional
            Output format: 'text', 'mpl' (matplotlib), 'latex', by default 'text'
        filename : str, optional
            Save to file if provided

        Returns
        -------
        str or None
            Text representation if output='text', else None
        """
        if output == "text":
            return str(self.circuit.draw(output="text"))
        else:
            fig = self.circuit.draw(output=output)
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches="tight")
            return None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantumCircuitBuilder(n_qubits={self.n_qubits}, "
            f"depth={self.depth()}, "
            f"n_params={self.num_parameters()})"
        )

    def __str__(self) -> str:
        """Pretty string representation."""
        return self.draw(output="text")


def create_bell_state(measure: bool = False) -> QuantumCircuitBuilder:
    """
    Create a Bell state (EPR pair) circuit.

    Creates the maximally entangled state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

    Parameters
    ----------
    measure : bool, optional
        Whether to add measurements, by default False

    Returns
    -------
    QuantumCircuitBuilder
        Circuit creating Bell state

    Examples
    --------
    >>> bell = create_bell_state()
    >>> sv = bell.get_statevector()
    >>> np.allclose(np.abs(sv)**2, [0.5, 0, 0, 0.5])
    True
    """
    builder = QuantumCircuitBuilder(n_qubits=2, name="Bell_state")
    builder.h(0).cnot(0, 1)

    if measure:
        builder.measure_all()

    return builder


def create_ghz_state(n_qubits: int, measure: bool = False) -> QuantumCircuitBuilder:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state.

    Creates the maximally entangled state (|00...0⟩ + |11...1⟩)/√2

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 2)
    measure : bool, optional
        Whether to add measurements, by default False

    Returns
    -------
    QuantumCircuitBuilder
        Circuit creating GHZ state
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    builder = QuantumCircuitBuilder(n_qubits=n_qubits, name=f"GHZ_{n_qubits}")
    builder.h(0)

    for i in range(n_qubits - 1):
        builder.cnot(i, i + 1)

    if measure:
        builder.measure_all()

    return builder


def create_w_state(n_qubits: int) -> QuantumCircuitBuilder:
    """
    Create a W state (another type of entangled state).

    Creates |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 2)

    Returns
    -------
    QuantumCircuitBuilder
        Circuit creating W state
    """
    if n_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")

    builder = QuantumCircuitBuilder(n_qubits=n_qubits, name=f"W_{n_qubits}")

    # W state preparation using recursive method
    # This is a simplified version for educational purposes
    builder.ry(2 * np.arccos(np.sqrt(1 / n_qubits)), 0)

    for i in range(n_qubits - 1):
        angle = 2 * np.arccos(np.sqrt(1 / (n_qubits - i)))
        builder.cnot(i, i + 1)
        builder.ry(angle, i + 1)

    return builder
