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

# Palette lives in qot_course.colors (single source of truth). These names are
# kept as backward-compatible aliases for the existing notebooks.
from qot_course.colors import COLORS, CMAP_COST, CMAP_PLAN, CMAP_DENSITY

SOURCE_COLOR = COLORS["source"]  # the distribution we have
TARGET_COLOR = COLORS["target"]  # the distribution we want
FLOW_COLOR = COLORS["flow"]      # mass in motion

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": COLORS["grid"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlesize": 14,
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
    source: np.ndarray,
    target: np.ndarray,
    source_positions: np.ndarray | None = None,
    target_positions: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar plot of source and target mass distributions over positions.

    When ``source_positions`` and ``target_positions`` are omitted or identical, the
    bars are drawn side-by-side over the shared integer grid (S1 behaviour). When the
    two distributions live on *different* position sets (the S8 Monge-fails case),
    each bar sits at its own position with the target partially transparent.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source_positions is None:
        source_positions = np.arange(len(source), dtype=float)
    if target_positions is None:
        target_positions = np.arange(len(target), dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    same_grid = (
        len(source) == len(target)
        and source_positions.shape == target_positions.shape
        and np.allclose(source_positions, target_positions)
    )
    if same_grid:
        ax.bar(
            source_positions - 0.18,
            source,
            width=0.36,
            color=SOURCE_COLOR,
            label="source — the pile we have",
            alpha=0.9,
        )
        ax.bar(
            source_positions + 0.18,
            target,
            width=0.36,
            color=TARGET_COLOR,
            label="target — the pile we want",
            alpha=0.9,
        )
        ax.set_xticks(source_positions)
    else:
        ax.bar(
            source_positions,
            source,
            width=0.32,
            color=SOURCE_COLOR,
            label="source",
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.bar(
            target_positions,
            target,
            width=0.32,
            color=TARGET_COLOR,
            label="target",
            alpha=0.65,
            edgecolor="white",
            linewidth=1.0,
        )
        all_positions = np.unique(
            np.concatenate([source_positions, target_positions])
        )
        ax.set_xticks(all_positions)
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
    source_positions: np.ndarray | None = None,
    target_positions: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "Optimal transport — who moves where (line width $\\propto$ mass)",
) -> plt.Figure:
    """Flow view: source (top row) and target (bottom row) as mass-sized dots, with
    one line per non-zero transport, its width and opacity proportional to the mass.

    Pass ``source_positions`` / ``target_positions`` when source and target live on
    *different* position sets (e.g. the S8 mass-splitting example with 1 source atom
    and 2 target atoms). When omitted, both rows use ``np.arange`` over their own
    lengths (S1 behaviour, backward-compatible).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    plan = np.asarray(plan, dtype=float)
    if source_positions is None:
        source_positions = np.arange(len(source), dtype=float)
    if target_positions is None:
        target_positions = np.arange(len(target), dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

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
                    [source_positions[i], target_positions[j]],
                    [1, 0],
                    color=FLOW_COLOR,
                    alpha=0.2 + 0.6 * mass / peak,
                    linewidth=1.0 + 7.0 * mass / peak,
                    solid_capstyle="round",
                    zorder=2,
                )
    ax.scatter(
        source_positions,
        np.ones(len(source)),
        s=3000 * source,
        color=SOURCE_COLOR,
        alpha=0.9,
        zorder=3,
        label="source",
    )
    ax.scatter(
        target_positions,
        np.zeros(len(target)),
        s=3000 * target,
        color=TARGET_COLOR,
        alpha=0.9,
        zorder=3,
        label="target",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["target", "source"])
    all_positions = np.unique(np.concatenate([source_positions, target_positions]))
    ax.set_xticks(all_positions)
    ax.set_xlabel("position")
    ax.set_title(title, pad=12)
    ax.grid(False)
    ax.margins(y=0.3)
    return fig


def plot_bloch(state, title: str = "") -> plt.Figure:
    """Plot a qubit state on the Bloch sphere (via Qiskit).

    Accepts either a pure-state ket (length-2 vector, placed on the surface) or
    a 2x2 density matrix (mixed states land inside the ball, the Bloch vector
    shrinking toward the centre as purity falls).
    """
    from qiskit.visualization import plot_bloch_vector

    arr = np.asarray(state, dtype=complex)
    if arr.shape == (2, 2):
        # Density matrix: r = (tr(rho X), tr(rho Y), tr(rho Z)); length < 1 when mixed.
        from qot_course.quantum.density import bloch_vector
    else:
        # Pure-state ket on the unit sphere.
        from qot_course.quantum.states import bloch_vector

    # Render onto our own 3D axes so Qiskit does not auto-close the figure
    # under an inline (Jupyter) backend. When given an ``ax``, plot_bloch_vector
    # draws in place and returns None instead of closing the figure (which is
    # what it does otherwise, via matplotlib_close_if_inline). Returning the
    # live figure lets plt.show() / inline auto-display render it as usual.
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    plot_bloch_vector(list(bloch_vector(arr)), title=title, ax=ax)
    return fig


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


def plot_density_matrix(
    rho, title: str = "", basis_labels: list[str] | None = None
) -> plt.Figure:
    """Show the real and imaginary parts of a density matrix as annotated heatmaps.

    Parameters
    ----------
    rho : array_like, shape (n, n)
        Density matrix to display.
    title : str
        Figure suptitle.
    basis_labels : list[str] | None
        Tick labels for the n rows/columns, in row order. Use these to name the
        basis a matrix is written in, e.g. ``["|00>", "|01>", "|10>", "|11>"]``
        for a two-qubit system. When ``None`` (default) the axes show integer
        indices 0..n-1 — appropriate when the basis is not a qubit tensor basis
        (e.g. QOT transport plans / Gibbs states of arbitrary dimension).
    """
    rho = np.asarray(rho, dtype=complex)
    n = rho.shape[0]
    if basis_labels is not None and len(basis_labels) != n:
        raise ValueError(
            f"basis_labels has {len(basis_labels)} entries but rho is {n}x{n}"
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, part, name in zip(axes, (rho.real, rho.imag), ("Re(rho)", "Im(rho)")):
        im = ax.imshow(part, cmap=CMAP_DENSITY, vmin=-1.0, vmax=1.0)
        ax.set_title(name, pad=10)
        ax.set_xticks(range(part.shape[1]))
        ax.set_yticks(range(part.shape[0]))
        if basis_labels is not None:
            ax.set_xticklabels(basis_labels)
            ax.set_yticklabels(basis_labels)
        ax.grid(False)
        for i in range(part.shape[0]):
            for j in range(part.shape[1]):
                ax.annotate(
                    f"{part[i, j]:.2f}",
                    (j, i),
                    ha="center",
                    va="center",
                    color=COLORS["text"],
                    fontsize=11,
                )
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Simplex plotting (S6 — information geometry)
# --------------------------------------------------------------------------- #
_SIMPLEX_VERTICES_2D = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
)
_SIMPLEX_PATH_PALETTE = [FLOW_COLOR, SOURCE_COLOR, TARGET_COLOR, COLORS["quantum"], COLORS["highlight"]]


def _draw_simplex_axes(
    ax: plt.Axes,
    vertex_labels: tuple[str, str, str] = (
        r"$\delta_1=(1,0,0)$",
        r"$\delta_2=(0,1,0)$",
        r"$\delta_3=(0,0,1)$",
    ),
) -> None:
    """Draw the empty 2-simplex triangle with labelled vertices on ``ax``."""
    v = _SIMPLEX_VERTICES_2D
    triangle = np.vstack([v, v[0:1]])
    ax.plot(triangle[:, 0], triangle[:, 1], color=COLORS["muted"], lw=1.5, zorder=1)
    label_offsets = ((-12, -10), (12, -10), (0, 10))
    label_ha = ("right", "left", "center")
    label_va = ("top", "top", "bottom")
    for vert, label, off, ha, va in zip(
        v, vertex_labels, label_offsets, label_ha, label_va
    ):
        ax.annotate(
            label,
            vert,
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=11,
            fontweight="bold",
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.18, np.sqrt(3.0) / 2.0 + 0.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_simplex_points(
    named_points: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
    title: str = "The 2-simplex — categorical distributions over three outcomes",
) -> plt.Figure:
    """Place labelled categorical distributions inside the 2-simplex triangle."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    else:
        fig = ax.figure
    _draw_simplex_axes(ax)
    palette = [SOURCE_COLOR, TARGET_COLOR, FLOW_COLOR, COLORS["quantum"], COLORS["highlight"], COLORS["positive"]]
    for k, (name, p) in enumerate(named_points.items()):
        xy = np.asarray(p, dtype=float).ravel() @ _SIMPLEX_VERTICES_2D
        ax.scatter(
            *xy,
            color=palette[k % len(palette)],
            s=160,
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            name,
            xy,
            xytext=(9, 6),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_title(title, pad=14)
    return fig


def plot_simplex_paths(
    named_paths: dict[str, np.ndarray],
    endpoints: dict[str, np.ndarray] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Paths on the 2-simplex",
) -> plt.Figure:
    """Plot one or more curves (each an array of 3-vectors) on the 2-simplex."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    else:
        fig = ax.figure
    _draw_simplex_axes(ax)
    for k, (name, path) in enumerate(named_paths.items()):
        xy = np.asarray(path, dtype=float) @ _SIMPLEX_VERTICES_2D
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=_SIMPLEX_PATH_PALETTE[k % len(_SIMPLEX_PATH_PALETTE)],
            lw=2.5,
            label=name,
            zorder=2,
        )
    if endpoints:
        for name, p in endpoints.items():
            xy = np.asarray(p, dtype=float).ravel() @ _SIMPLEX_VERTICES_2D
            ax.scatter(
                *xy,
                color=COLORS["text"],
                s=80,
                zorder=4,
                edgecolor="white",
                linewidth=1.5,
            )
            ax.annotate(
                name,
                xy,
                xytext=(9, 6),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
            )
    ax.set_title(title, pad=14)
    if named_paths:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=len(named_paths),
            frameon=False,
        )
    return fig


def plot_interpolation_panel(
    support: np.ndarray,
    times: list[float],
    paths_by_label: dict[str, list[np.ndarray]],
    titles: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot one row per labelled interpolation (each a list of distributions over time).

    Designed for the S6 "two geometries" comparison: pass e.g. a ``"mixture"`` row and
    a ``"Wasserstein"`` row, each with the same set of time snapshots, on a shared
    1-D ``support`` --- the colour evolves from cold (t=0) to warm (t=1) along the path.
    """
    n_rows = len(paths_by_label)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    cmap = plt.colormaps["viridis"]
    for ax, (label, snapshots) in zip(axes, paths_by_label.items()):
        for k, (t, pt) in enumerate(zip(times, snapshots)):
            color = cmap(k / max(1, len(times) - 1))
            ax.plot(support, pt, color=color, lw=2.0, label=f"t={t:.2f}")
        ax.set_ylabel("probability mass")
        ax.set_title(
            (titles or {}).get(label, label), pad=10
        )
        ax.legend(loc="upper right", ncol=len(times), fontsize=9)
    axes[-1].set_xlabel("position  $x$")
    fig.tight_layout()
    return fig
