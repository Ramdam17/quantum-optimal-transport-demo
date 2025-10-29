"""
Quantum optimal transport visualization module.

This module provides visualization tools for quantum circuits, quantum states,
optimization convergence, and comparison between classical and quantum methods.

Functions
---------
plot_quantum_circuit
    Visualize quantum circuit diagram
plot_convergence
    Plot optimization convergence history
plot_cost_landscape
    Visualize cost function landscape
plot_quantum_vs_classical
    Compare quantum and classical OT results
plot_bloch_sphere
    Visualize single-qubit state on Bloch sphere
plot_statevector_bars
    Bar plot of statevector amplitudes
plot_probability_distribution
    Plot probability distribution from quantum state

Examples
--------
>>> from src.visualization.quantum_plots import plot_convergence
>>> import numpy as np
>>>
>>> history = [10.0, 8.0, 6.0, 5.0, 4.5]
>>> plot_convergence(history, save_path='convergence.png')

Notes
-----
This module uses matplotlib for static plots and can optionally use
qiskit visualization tools for circuit diagrams.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_quantum_circuit(
    circuit: Any,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    style: str = "mpl",
) -> Optional[Figure]:
    """
    Plot quantum circuit diagram.

    Parameters
    ----------
    circuit : Any
        QuantumCircuit or QuantumCircuitBuilder instance
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    style : str, optional
        Drawing style ('mpl', 'text', 'latex'), by default 'mpl'

    Returns
    -------
    Optional[Figure]
        Matplotlib figure if using 'mpl' style, None otherwise

    Examples
    --------
    >>> from src.quantum.circuits import QuantumCircuitBuilder
    >>> circuit = QuantumCircuitBuilder(3)
    >>> circuit.h(0).cnot(0, 1).cnot(1, 2)
    >>> plot_quantum_circuit(circuit.circuit, save_path='ghz_circuit.png')
    """
    try:
        from qiskit.visualization import circuit_drawer

        # Get underlying circuit if QuantumCircuitBuilder
        qc = circuit.circuit if hasattr(circuit, "circuit") else circuit

        # Draw circuit
        if style == "mpl":
            fig = circuit_drawer(qc, output="mpl", plot_barriers=False, fold=-1)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved circuit diagram to {save_path}")

            if show:
                plt.show()
            else:
                plt.close()

            return fig
        else:
            # Text or LaTeX output
            print(circuit_drawer(qc, output=style))
            return None

    except ImportError:
        logger.warning("Qiskit visualization not available. Using text output.")
        print(str(circuit))
        return None


def plot_convergence(
    convergence_history: Union[List[float], np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Cost",
    log_scale: bool = False,
) -> Figure:
    """
    Plot optimization convergence history.

    Parameters
    ----------
    convergence_history : Union[List[float], np.ndarray]
        Array of cost values during optimization
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default "Optimization Convergence"
    xlabel : str, optional
        X-axis label, by default "Iteration"
    ylabel : str, optional
        Y-axis label, by default "Cost"
    log_scale : bool, optional
        Use logarithmic scale for y-axis, by default False

    Returns
    -------
    Figure
        Matplotlib figure

    Examples
    --------
    >>> history = [10.0, 8.0, 6.5, 5.8, 5.5, 5.4]
    >>> fig = plot_convergence(history, title='VQE Convergence')
    """
    fig, ax = plt.subplots(figsize=figsize)

    history = np.asarray(convergence_history)
    iterations = np.arange(len(history))

    # Handle empty history
    if len(history) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
    else:
        # Plot convergence
        ax.plot(iterations, history, "b-", linewidth=2, label="Cost")
        ax.scatter(iterations, history, c="blue", s=30, alpha=0.6)

        # Mark best value
        best_idx = np.argmin(history)
        ax.scatter(
            best_idx,
            history[best_idx],
            c="red",
            s=100,
            marker="*",
            label=f"Best: {history[best_idx]:.4f}",
            zorder=5,
        )

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if len(history) > 0:
        ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved convergence plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_cost_landscape(
    param_ranges: List[Tuple[float, float]],
    cost_function: callable,
    optimal_params: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    resolution: int = 30,
    title: str = "Cost Landscape",
) -> Figure:
    """
    Plot 2D cost function landscape.

    Parameters
    ----------
    param_ranges : List[Tuple[float, float]]
        Ranges for first two parameters [(min1, max1), (min2, max2)]
    cost_function : callable
        Function that takes parameters and returns cost
    optimal_params : Optional[np.ndarray], optional
        Optimal parameters to mark on plot, by default None
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    resolution : int, optional
        Grid resolution, by default 30
    title : str, optional
        Plot title, by default "Cost Landscape"

    Returns
    -------
    Figure
        Matplotlib figure

    Examples
    --------
    >>> def cost_fn(params):
    ...     return (params[0] - 1)**2 + (params[1] + 2)**2
    >>> ranges = [(0, 2), (-3, -1)]
    >>> fig = plot_cost_landscape(ranges, cost_fn)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create grid
    x = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
    y = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute costs
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            params = np.array([X[i, j], Y[i, j]])
            Z[i, j] = cost_function(params)

    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.8)
    contour_lines = ax.contour(
        X, Y, Z, levels=10, colors="black", alpha=0.3, linewidths=0.5
    )
    ax.clabel(contour_lines, inline=True, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Cost", fontsize=12)

    # Mark optimal point
    if optimal_params is not None:
        ax.scatter(
            optimal_params[0],
            optimal_params[1],
            c="red",
            s=200,
            marker="*",
            edgecolors="white",
            linewidths=2,
            label="Optimal",
            zorder=5,
        )
        ax.legend(loc="best", fontsize=10)

    # Formatting
    ax.set_xlabel("Parameter 1", fontsize=12)
    ax.set_ylabel("Parameter 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cost landscape to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_quantum_vs_classical(
    classical_result: Dict[str, Any],
    quantum_result: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """
    Compare quantum and classical OT results.

    Parameters
    ----------
    classical_result : Dict[str, Any]
        Classical OT result dictionary (must contain 'cost')
    quantum_result : Dict[str, Any]
        Quantum OT result dictionary (must contain 'cost' and 'convergence_history')
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 5)

    Returns
    -------
    Figure
        Matplotlib figure with two subplots

    Examples
    --------
    >>> classical = {'cost': 2.5}
    >>> quantum = {'cost': 2.8, 'convergence_history': [10, 8, 5, 3, 2.8]}
    >>> fig = plot_quantum_vs_classical(classical, quantum)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Cost comparison bar plot
    methods = ["Classical", "Quantum"]
    costs = [classical_result["cost"], quantum_result["cost"]]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax1.bar(methods, costs, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("OT Cost", fontsize=12)
    ax1.set_title("Cost Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Quantum convergence plot
    if "convergence_history" in quantum_result:
        history = quantum_result["convergence_history"]
        iterations = np.arange(len(history))

        ax2.plot(iterations, history, "o-", color="#A23B72", linewidth=2, markersize=5)
        ax2.axhline(
            y=classical_result["cost"],
            color="#2E86AB",
            linestyle="--",
            linewidth=2,
            label="Classical Cost",
        )

        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Cost", fontsize=12)
        ax2.set_title(
            "Quantum Optimization Convergence", fontsize=14, fontweight="bold"
        )
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_statevector_bars(
    statevector: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Quantum State Amplitudes",
    max_states: int = 16,
) -> Figure:
    """
    Plot statevector amplitudes as bar charts.

    Parameters
    ----------
    statevector : np.ndarray
        Quantum statevector (complex amplitudes)
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 5)
    title : str, optional
        Plot title, by default "Quantum State Amplitudes"
    max_states : int, optional
        Maximum number of basis states to display, by default 16

    Returns
    -------
    Figure
        Matplotlib figure

    Examples
    --------
    >>> state = np.array([0.7+0j, 0.3+0j, 0, 0])
    >>> fig = plot_statevector_bars(state)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    statevector = np.asarray(statevector)
    n_states = min(len(statevector), max_states)

    # Compute amplitudes and phases
    amplitudes = np.abs(statevector[:n_states])
    phases = np.angle(statevector[:n_states])

    # Basis state labels
    n_qubits = int(np.log2(len(statevector)))
    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    x_pos = np.arange(n_states)

    # Plot amplitudes
    ax1.bar(x_pos, amplitudes, color="#2E86AB", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Basis State", fontsize=12)
    ax1.set_ylabel("Amplitude |ψ|", fontsize=12)
    ax1.set_title("Amplitudes", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot phases
    colors = plt.cm.hsv(phases / (2 * np.pi))
    ax2.bar(x_pos, phases, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Basis State", fontsize=12)
    ax2.set_ylabel("Phase (radians)", fontsize=12)
    ax2.set_title("Phases", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved statevector plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_probability_distribution(
    probabilities: Union[Dict[str, float], np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Probability Distribution",
    max_states: int = 16,
    threshold: float = 0.01,
) -> Figure:
    """
    Plot probability distribution from quantum measurements.

    Parameters
    ----------
    probabilities : Union[Dict[str, float], np.ndarray]
        Probability distribution (dict or array)
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default "Probability Distribution"
    max_states : int, optional
        Maximum number of states to display, by default 16
    threshold : float, optional
        Minimum probability to display, by default 0.01

    Returns
    -------
    Figure
        Matplotlib figure

    Examples
    --------
    >>> probs = {'00': 0.5, '01': 0.3, '10': 0.2}
    >>> fig = plot_probability_distribution(probs)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to dict if array
    if isinstance(probabilities, np.ndarray):
        n_qubits = int(np.log2(len(probabilities)))
        probabilities = {
            format(i, f"0{n_qubits}b"): p
            for i, p in enumerate(probabilities)
            if p > threshold
        }

    # Filter and sort
    probs_filtered = {k: v for k, v in probabilities.items() if v > threshold}
    probs_sorted = dict(
        sorted(probs_filtered.items(), key=lambda x: x[1], reverse=True)
    )

    # Limit number of states
    if len(probs_sorted) > max_states:
        items = list(probs_sorted.items())[:max_states]
        probs_sorted = dict(items)

    # Plot
    states = list(probs_sorted.keys())
    probs = list(probs_sorted.values())
    x_pos = np.arange(len(states))

    bars = ax.bar(x_pos, probs, color="#A23B72", alpha=0.7, edgecolor="black")

    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Formatting
    ax.set_xlabel("Basis State", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(states, rotation=45, ha="right")
    ax.set_ylim(0, max(probs) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved probability distribution to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_multiple_convergences(
    histories: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Optimization Comparison",
    log_scale: bool = False,
) -> Figure:
    """
    Plot multiple optimization convergence histories for comparison.

    Parameters
    ----------
    histories : Dict[str, np.ndarray]
        Dictionary mapping method names to convergence arrays
    save_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    show : bool, optional
        Whether to display the plot, by default True
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    title : str, optional
        Plot title, by default "Optimization Comparison"
    log_scale : bool, optional
        Use logarithmic scale, by default False

    Returns
    -------
    Figure
        Matplotlib figure

    Examples
    --------
    >>> histories = {
    ...     'VQE': np.array([10, 8, 6, 5, 4.5]),
    ...     'QAOA': np.array([12, 9, 7, 6.5, 6])
    ... }
    >>> fig = plot_multiple_convergences(histories)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (method, history), color in zip(histories.items(), colors):
        history = np.asarray(history)
        iterations = np.arange(len(history))

        ax.plot(
            iterations,
            history,
            "o-",
            label=method,
            color=color,
            linewidth=2,
            markersize=5,
        )

    # Formatting
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved multiple convergences plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig
