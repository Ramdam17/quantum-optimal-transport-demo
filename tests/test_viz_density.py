import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qot_course import viz
from qot_course.quantum.density import density_matrix
from qot_course.quantum.states import KET_PLUS


def test_plot_density_matrix_returns_figure():
    rho = density_matrix(KET_PLUS)
    assert isinstance(viz.plot_density_matrix(rho, title="|+>"), Figure)
