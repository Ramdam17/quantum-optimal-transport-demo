import matplotlib

matplotlib.use("Agg")  # headless backend; set before pyplot is imported via intro

import numpy as np
from matplotlib.figure import Figure

from qot_course.intro import check_environment, plot_1d_transport


def test_check_environment_reports_core_packages():
    env = check_environment()
    assert env["numpy"] is not None  # numpy is installed
    assert "qiskit" in env  # key reported even if value could be a version string


def test_plot_1d_transport_returns_figure():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.5, 0.5])
    fig = plot_1d_transport(source, target)
    assert isinstance(fig, Figure)


def test_plot_1d_transport_with_plan_runs():
    source = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])
    plan = np.array([[0.0, 1.0], [0.0, 0.0]])
    fig = plot_1d_transport(source, target, transport_plan=plan)
    assert isinstance(fig, Figure)
