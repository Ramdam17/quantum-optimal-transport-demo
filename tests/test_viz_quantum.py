"""Check what the Bloch-sphere and counts figures draw."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qot_course import viz
from qot_course.quantum.density import maximally_mixed
from qot_course.quantum.states import KET_PLUS


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("state", [KET_PLUS, maximally_mixed(2)], ids=["ket", "rho"])
def test_plot_bloch_draws_on_one_3d_axes_for_a_ket_and_for_a_density_matrix(state):
    fig = viz.plot_bloch(state, title="a state")
    (ax,) = fig.axes
    assert ax.name == "3d"
    # The figure stays open: Qiskit closes its own figure under an inline backend.
    assert plt.fignum_exists(fig.number)


def test_plot_counts_draws_one_bar_per_outcome_with_its_count():
    counts = {"1": 212, "0": 300}
    ax = viz.plot_counts(counts).axes[0]
    (bars,) = ax.containers
    # Outcomes are sorted, so the bar of "0" comes first whatever the dictionary order.
    np.testing.assert_array_equal([bar.get_height() for bar in bars], [300, 212])
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1"]
    assert [text.get_text() for text in ax.texts] == ["300", "212"]
