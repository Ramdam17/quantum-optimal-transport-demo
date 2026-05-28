import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from qot_course import viz
from qot_course.geometry.info_geometry import (
    fisher_rao_geodesic,
    mixture_interpolation,
    wasserstein_interpolation_1d,
)


def test_plot_simplex_points_returns_figure():
    fig = viz.plot_simplex_points(
        {"uniform": np.array([1 / 3, 1 / 3, 1 / 3]), "skewed": np.array([0.7, 0.2, 0.1])}
    )
    assert isinstance(fig, Figure)


def test_plot_simplex_paths_returns_figure():
    p = np.array([0.8, 0.1, 0.1])
    q = np.array([0.1, 0.8, 0.1])
    ts = np.linspace(0.0, 1.0, 30)
    arc = np.array([fisher_rao_geodesic(p, q, t) for t in ts])
    line = np.array([mixture_interpolation(p, q, t) for t in ts])
    fig = viz.plot_simplex_paths(
        {"FR geodesic": arc, "mixture": line},
        endpoints={"p": p, "q": q},
    )
    assert isinstance(fig, Figure)


def _normalized_bump(support: np.ndarray, center: float) -> np.ndarray:
    raw = np.exp(-0.5 * ((support - center) / 1.0) ** 2)
    return raw / raw.sum()


def test_plot_interpolation_panel_returns_figure():
    support = np.linspace(0.0, 10.0, 50)
    p = _normalized_bump(support, 2.0)
    q = _normalized_bump(support, 8.0)
    ts = [0.0, 0.5, 1.0]
    fig = viz.plot_interpolation_panel(
        support,
        ts,
        {
            "mix": [mixture_interpolation(p, q, t) for t in ts],
            "w2": [wasserstein_interpolation_1d(p, q, support, t) for t in ts],
        },
        titles={"mix": "Mixture (information)", "w2": "Wasserstein (transport)"},
    )
    assert isinstance(fig, Figure)
