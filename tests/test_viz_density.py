"""Check that the density-matrix figure shows the real and imaginary parts it was given."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qot_course import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# |+i><+i|: real populations, purely imaginary coherences of opposite signs.
RHO_PLUS_I = np.array([[0.5, -0.5j], [0.5j, 0.5]])


def test_plot_density_matrix_shows_real_and_imaginary_parts():
    fig = viz.plot_density_matrix(RHO_PLUS_I, title="|+i>", basis_labels=["0", "1"])
    real_ax, imag_ax = fig.axes[:2]  # the two colour bars come after
    np.testing.assert_array_equal(real_ax.images[0].get_array(), RHO_PLUS_I.real)
    np.testing.assert_array_equal(imag_ax.images[0].get_array(), RHO_PLUS_I.imag)
    # A fixed colour scale, so that two figures can be compared by eye.
    assert real_ax.images[0].get_clim() == (-1.0, 1.0)
    assert imag_ax.images[0].get_clim() == (-1.0, 1.0)
    assert fig.get_suptitle() == "|+i>"
    assert [tick.get_text() for tick in real_ax.get_xticklabels()] == ["0", "1"]


def test_plot_density_matrix_writes_each_entry_in_its_cell():
    fig = viz.plot_density_matrix(RHO_PLUS_I)
    imag_ax = fig.axes[1]
    written = {tuple(text.xy): text.get_text() for text in imag_ax.texts}
    # xy is (column, row): the entry of row 0, column 1 is -0.5.
    assert written == {(0, 0): "0.00", (1, 0): "-0.50", (0, 1): "0.50", (1, 1): "0.00"}


def test_plot_density_matrix_rejects_a_wrong_number_of_labels():
    with pytest.raises(ValueError, match="basis_labels"):
        viz.plot_density_matrix(RHO_PLUS_I, basis_labels=["0", "1", "2"])
