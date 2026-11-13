"""Check what the transport figures draw, not only that a figure comes back.

Each test reads the drawn artists (bars, image, lines, markers) and compares them with the
arrays handed to the helper, so a helper that plots the wrong array, swaps the two
distributions or drops a transport line fails here.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend; set before pyplot is imported via viz

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qot_course import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bar_heights(container) -> list[float]:
    return [bar.get_height() for bar in container]


def _bar_centres(container) -> list[float]:
    return [bar.get_x() + bar.get_width() / 2 for bar in container]


def test_use_course_style_sets_the_course_rcparams():
    viz.use_course_style()
    for key, value in viz._STYLE.items():
        assert matplotlib.rcParams[key] == matplotlib.rcParams.validate[key](value)


def test_plot_distributions_draws_source_then_target_on_a_shared_grid():
    source = np.array([0.5, 0.3, 0.2])
    target = np.array([0.2, 0.3, 0.5])
    (ax,) = viz.plot_distributions(source, target).axes
    drawn_source, drawn_target = ax.containers
    np.testing.assert_allclose(_bar_heights(drawn_source), source)
    np.testing.assert_allclose(_bar_heights(drawn_target), target)
    # The two bars of a position sit side by side, the source on the left.
    assert np.all(
        np.array(_bar_centres(drawn_source)) < np.array(_bar_centres(drawn_target))
    )
    np.testing.assert_allclose(ax.get_xticks(), [0.0, 1.0, 2.0])


def test_plot_distributions_puts_each_bar_at_its_own_position():
    source, source_positions = np.array([0.5, 0.5]), np.array([0.0, 4.0])
    target, target_positions = np.full(3, 1 / 3), np.array([1.0, 2.0, 3.0])
    fig = viz.plot_distributions(source, target, source_positions, target_positions)
    drawn_source, drawn_target = fig.axes[0].containers
    np.testing.assert_allclose(_bar_centres(drawn_source), source_positions)
    np.testing.assert_allclose(_bar_heights(drawn_source), source)
    np.testing.assert_allclose(_bar_centres(drawn_target), target_positions)
    np.testing.assert_allclose(_bar_heights(drawn_target), target)
    np.testing.assert_allclose(fig.axes[0].get_xticks(), [0.0, 1.0, 2.0, 3.0, 4.0])


def test_plot_cost_matrix_shows_the_cost_it_was_given():
    cost = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    (image,) = viz.plot_cost_matrix(cost).axes[0].images
    np.testing.assert_array_equal(image.get_array(), cost)
    assert image.origin == "lower"  # row i = source position i, read upwards


def test_plot_transport_plan_shows_the_plan_it_was_given():
    plan = np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.0], [0.1, 0.0, 0.1]])
    ax = viz.plot_transport_plan(plan, title="Optimal plan").axes[0]
    (image,) = ax.images
    np.testing.assert_array_equal(image.get_array(), plan)
    assert image.origin == "lower"
    assert ax.get_title() == "Optimal plan"


def test_plot_transport_arrows_draws_one_line_per_moved_mass():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.25, 0.75])
    plan = np.array([[0.0, 0.25, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    ax = viz.plot_transport_arrows(source, target, plan).axes[0]

    # One line per positive entry, from (source position, 1) down to (target position, 0).
    drawn = {
        (line.get_xdata()[0], line.get_xdata()[1]): line.get_linewidth()
        for line in ax.lines
    }
    assert set(drawn) == {(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)}
    for line in ax.lines:
        np.testing.assert_array_equal(line.get_ydata(), [1, 0])
    # The line carrying the most mass is the widest; equal masses get equal widths.
    assert drawn[(1.0, 2.0)] > drawn[(0.0, 1.0)] == drawn[(0.0, 2.0)]

    # Marker areas are proportional to the masses, the source row above the target row.
    source_markers, target_markers = ax.collections
    np.testing.assert_allclose(source_markers.get_sizes(), 3000 * source)
    np.testing.assert_allclose(target_markers.get_sizes(), 3000 * target)
    np.testing.assert_allclose(source_markers.get_offsets()[:, 1], 1.0)
    np.testing.assert_allclose(target_markers.get_offsets()[:, 1], 0.0)
