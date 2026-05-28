import matplotlib

matplotlib.use("Agg")  # headless backend; set before pyplot is imported via viz

import numpy as np
from matplotlib.figure import Figure

from qot_course import viz


def test_use_course_style_runs():
    viz.use_course_style()  # should not raise


def test_plot_distributions_returns_figure():
    source = np.array([0.5, 0.3, 0.2])
    target = np.array([0.2, 0.3, 0.5])
    assert isinstance(viz.plot_distributions(source, target), Figure)


def test_plot_cost_matrix_returns_figure():
    cost = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    assert isinstance(viz.plot_cost_matrix(cost), Figure)


def test_plot_transport_plan_returns_figure():
    plan = np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]])
    assert isinstance(viz.plot_transport_plan(plan, title="Optimal plan"), Figure)


def test_plot_transport_arrows_returns_figure():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.5, 0.5])
    plan = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    assert isinstance(viz.plot_transport_arrows(source, target, plan), Figure)
