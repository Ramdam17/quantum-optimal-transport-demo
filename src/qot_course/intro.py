"""Session 1 helpers: environment check and a 1D optimal-transport visual."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import matplotlib.pyplot as plt
import numpy as np

_KEY_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pot",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "cvxpy",
]


def check_environment() -> dict[str, str | None]:
    """Return installed versions of the course's key packages.

    Maps each package name to its version string, or ``None`` if missing.
    """
    versions: dict[str, str | None] = {}
    for package in _KEY_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def plot_1d_transport(
    source: np.ndarray,
    target: np.ndarray,
    transport_plan: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot two 1D mass distributions and, optionally, a transport plan as arrows.

    Parameters
    ----------
    source, target : np.ndarray
        1D mass vectors over integer positions ``0..n-1``.
    transport_plan : np.ndarray, optional
        ``(n, n)`` plan where ``P[i, j]`` is the mass moved from ``i`` to ``j``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    x = np.arange(len(source))
    ax.bar(x - 0.15, source, width=0.3, label="source", alpha=0.8)
    ax.bar(x + 0.15, target, width=0.3, label="target", alpha=0.8)

    if transport_plan is not None:
        plan = np.asarray(transport_plan, dtype=float)
        peak = plan.max() if plan.max() > 0 else 1.0
        for i in range(plan.shape[0]):
            for j in range(plan.shape[1]):
                if plan[i, j] > 1e-9:
                    ax.annotate(
                        "",
                        xy=(j + 0.15, target[j]),
                        xytext=(i - 0.15, source[i]),
                        arrowprops={
                            "arrowstyle": "->",
                            "alpha": 0.2 + 0.6 * plan[i, j] / peak,
                        },
                    )

    ax.set_xlabel("position")
    ax.set_ylabel("mass")
    ax.legend()
    return fig
