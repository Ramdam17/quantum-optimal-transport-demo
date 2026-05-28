from qot_course.intro import check_environment


def test_check_environment_reports_core_packages():
    env = check_environment()
    assert env["numpy"] is not None  # numpy is installed
    assert "qiskit" in env  # key present even if the value is a version string
    assert set(env) >= {
        "pot",
        "cvxpy",
        "matplotlib",
    }  # all course core packages reported
