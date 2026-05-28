import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from qot_course import viz
from qot_course.quantum.states import KET_PLUS, sample_counts


def test_plot_bloch_returns_figure():
    assert isinstance(viz.plot_bloch(KET_PLUS, title="|+>"), Figure)


def test_plot_counts_returns_figure():
    counts = sample_counts(KET_PLUS, shots=512, seed=0)
    assert isinstance(viz.plot_counts(counts), Figure)
