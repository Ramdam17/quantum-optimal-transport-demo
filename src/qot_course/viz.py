"""Course plotting style and optimal-transport visualizations.

A clean, high-contrast light style with vibrant accents, so figures read well
both in notebooks and in the LaTeX PDF summaries. Each optimal-transport object
gets its own, fully labelled figure (distributions, cost matrix, transport plan,
mass flow) — never an unlabelled tangle.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Vibrant, consistent palette used across the whole course.
SOURCE_COLOR = "#7c3aed"  # violet — the distribution we have
TARGET_COLOR = "#f59e0b"  # amber  — the distribution we want
FLOW_COLOR = "#10b981"  # emerald — mass in motion
CMAP_COST = "magma"  # cost matrices
CMAP_PLAN = "plasma"  # transport plans (mass)

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.6,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def use_course_style() -> None:
    """Apply the course's matplotlib style globally (idempotent)."""
    mpl.rcParams.update(_STYLE)


def plot_distributions(
    source: np.ndarray, target: np.ndarray, ax: plt.Axes | None = None
) -> plt.Figure:
    """Bar plot of the source and target mass distributions over positions."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure
    x = np.arange(len(source))
    ax.bar(
        x - 0.18,
        source,
        width=0.36,
        color=SOURCE_COLOR,
        label="source — the pile we have",
        alpha=0.9,
    )
    ax.bar(
        x + 0.18,
        target,
        width=0.36,
        color=TARGET_COLOR,
        label="target — the pile we want",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xlabel("position")
    ax.set_ylabel("probability mass")
    ax.set_title("Two distributions over positions", pad=12)
    ax.legend()
    return fig


def plot_cost_matrix(cost: np.ndarray, ax: plt.Axes | None = None) -> plt.Figure:
    """Heatmap of the ground cost ``c(i, j)`` of moving one unit from ``i`` to ``j``."""
    cost = np.asarray(cost, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5))
    else:
        fig = ax.figure
    im = ax.imshow(cost, cmap=CMAP_COST, origin="lower")
    ax.set_xlabel("target position $j$")
    ax.set_ylabel("source position $i$")
    ax.set_title("Cost matrix  $c(i, j) = (i - j)^2$", pad=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="cost to move one unit", shrink=0.85)
    return fig


def plot_transport_plan(
    plan: np.ndarray, ax: plt.Axes | None = None, title: str = "Transport plan"
) -> plt.Figure:
    """Heatmap of a transport plan ``P[i, j]`` = mass moved from ``i`` to ``j``."""
    plan = np.asarray(plan, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5))
    else:
        fig = ax.figure
    im = ax.imshow(plan, cmap=CMAP_PLAN, origin="lower")
    ax.set_xlabel("target position $j$")
    ax.set_ylabel("source position $i$")
    ax.set_title(title, pad=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mass moved  $i \\to j$", shrink=0.85)
    return fig


def plot_transport_arrows(
    source: np.ndarray,
    target: np.ndarray,
    plan: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Flow view: source (top) and target (bottom) as mass-sized dots, with one
    line per non-zero transport, its width and opacity proportional to the mass."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    plan = np.asarray(plan, dtype=float)
    n = len(source)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure

    peak = plan.max() if plan.max() > 0 else 1.0
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            mass = plan[i, j]
            if mass > 1e-9:
                ax.plot(
                    [i, j],
                    [1, 0],
                    color=FLOW_COLOR,
                    alpha=0.2 + 0.6 * mass / peak,
                    linewidth=1.0 + 7.0 * mass / peak,
                    solid_capstyle="round",
                    zorder=2,
                )
    ax.scatter(
        np.arange(n),
        np.ones(n),
        s=3000 * source,
        color=SOURCE_COLOR,
        alpha=0.9,
        zorder=3,
        label="source",
    )
    ax.scatter(
        np.arange(n),
        np.zeros(n),
        s=3000 * target,
        color=TARGET_COLOR,
        alpha=0.9,
        zorder=3,
        label="target",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["target", "source"])
    ax.set_xticks(np.arange(n))
    ax.set_xlabel("position")
    ax.set_title(
        "Optimal transport — who moves where (line width $\\propto$ mass)", pad=12
    )
    ax.grid(False)
    ax.margins(y=0.3)
    return fig


def plot_bloch(state, title: str = "") -> plt.Figure:
    """Plot a pure qubit state on the Bloch sphere (via Qiskit)."""
    from qiskit.visualization import plot_bloch_vector

    from qot_course.quantum.states import bloch_vector

    return plot_bloch_vector(list(bloch_vector(state)), title=title)


def plot_counts(counts: dict[str, int], ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart of measurement counts (computational basis)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure
    labels = sorted(counts)
    values = [counts[k] for k in labels]
    ax.bar(labels, values, color=[SOURCE_COLOR, TARGET_COLOR][: len(labels)], alpha=0.9)
    ax.set_xlabel("measurement outcome")
    ax.set_ylabel("counts")
    ax.set_title("Measurement outcomes", pad=12)
    for label, value in zip(labels, values):
        ax.annotate(str(value), (label, value), ha="center", va="bottom", fontsize=11)
    return fig


def plot_density_matrix(rho, title: str = "") -> plt.Figure:
    """Show the real and imaginary parts of a density matrix as annotated heatmaps."""
    rho = np.asarray(rho, dtype=complex)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, part, name in zip(axes, (rho.real, rho.imag), ("Re(rho)", "Im(rho)")):
        im = ax.imshow(part, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(name, pad=10)
        ax.set_xticks(range(part.shape[1]))
        ax.set_yticks(range(part.shape[0]))
        ax.grid(False)
        for i in range(part.shape[0]):
            for j in range(part.shape[1]):
                ax.annotate(
                    f"{part[i, j]:.2f}",
                    (j, i),
                    ha="center",
                    va="center",
                    color="#0d1117",
                    fontsize=11,
                )
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig
