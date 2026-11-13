"""Check where the simplex and interpolation figures put the points they are given."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qot_course import viz
from qot_course.geometry.info_geometry import (
    fisher_rao_geodesic,
    mixture_interpolation,
    wasserstein_interpolation_1d,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# Corners of the drawn triangle, in the order of the three outcomes.
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def test_plot_simplex_points_puts_each_distribution_at_its_barycentre():
    points = {
        "first outcome": np.array([1.0, 0.0, 0.0]),
        "third outcome": np.array([0.0, 0.0, 1.0]),
        "uniform": np.array([1 / 3, 1 / 3, 1 / 3]),
    }
    ax = viz.plot_simplex_points(points).axes[0]
    drawn = np.vstack([markers.get_offsets() for markers in ax.collections])
    expected = [CORNERS[0], CORNERS[2], CORNERS.mean(axis=0)]
    np.testing.assert_allclose(drawn, expected, atol=1e-12)
    # The three corner labels come first, then one label per point, in order.
    assert [text.get_text() for text in ax.texts][3:] == list(points)


def test_plot_simplex_paths_draws_each_path_and_its_endpoints():
    p = np.array([0.8, 0.1, 0.1])
    q = np.array([0.1, 0.8, 0.1])
    ts = np.linspace(0.0, 1.0, 30)
    arc = np.array([fisher_rao_geodesic(p, q, t) for t in ts])
    line = np.array([mixture_interpolation(p, q, t) for t in ts])
    ax = viz.plot_simplex_paths(
        {"FR geodesic": arc, "mixture": line}, endpoints={"p": p, "q": q}
    ).axes[0]

    outline, drawn_arc, drawn_line = ax.lines
    np.testing.assert_allclose(outline.get_xydata()[:3], CORNERS)
    np.testing.assert_allclose(drawn_arc.get_xydata(), arc @ CORNERS)
    np.testing.assert_allclose(drawn_line.get_xydata(), line @ CORNERS)
    # The mixture path is a straight segment; the Fisher-Rao geodesic bows away from it.
    midpoint = 0.5 * (p + q) @ CORNERS
    np.testing.assert_allclose(drawn_line.get_xydata()[[0, -1]].mean(axis=0), midpoint)
    assert np.abs(drawn_arc.get_xydata() - drawn_line.get_xydata()).max() > 1e-2

    endpoints = np.vstack([markers.get_offsets() for markers in ax.collections])
    np.testing.assert_allclose(endpoints, [p @ CORNERS, q @ CORNERS])
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["FR geodesic", "mixture"]


def _normalized_bump(support: np.ndarray, center: float) -> np.ndarray:
    raw = np.exp(-0.5 * ((support - center) / 1.0) ** 2)
    return raw / raw.sum()


def test_plot_interpolation_panel_draws_one_row_per_path_and_one_curve_per_time():
    support = np.linspace(0.0, 10.0, 50)
    p = _normalized_bump(support, 2.0)
    q = _normalized_bump(support, 8.0)
    ts = [0.0, 0.5, 1.0]
    paths = {
        "mix": [mixture_interpolation(p, q, t) for t in ts],
        "w2": [wasserstein_interpolation_1d(p, q, support, t) for t in ts],
    }
    titles = {"mix": "Mixture (information)", "w2": "Wasserstein (transport)"}
    fig = viz.plot_interpolation_panel(support, ts, paths, titles=titles)

    assert [ax.get_title() for ax in fig.axes] == list(titles.values())
    for ax, snapshots in zip(fig.axes, paths.values()):
        assert len(ax.lines) == len(ts)
        for curve, snapshot in zip(ax.lines, snapshots):
            np.testing.assert_array_equal(curve.get_xdata(), support)
            np.testing.assert_allclose(curve.get_ydata(), snapshot)
        legend = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend == ["t=0.00", "t=0.50", "t=1.00"]
