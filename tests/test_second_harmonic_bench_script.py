import csv
import json

import pytest

from scripts.run_second_harmonic_bench import main


def _rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    """Output directory of one quick run with the default configuration.

    The quick run takes several seconds, so the tests that only read its output
    share it; a test that changes the configuration makes its own run.
    """
    out = tmp_path_factory.mktemp("quick_run")
    assert main(["--quick", "--out", str(out)]) == 0
    return out


def test_quick_run_writes_every_result_file(quick_run):
    for name in (
        "map.csv",
        "freedom.csv",
        "continuity.json",
        "timestep_check.csv",
        "comparisons.csv",
        "summary.csv",
        "reliability.csv",
        "run_metadata.json",
    ):
        assert (quick_run / name).stat().st_size > 0, name
    assert len(_rows(quick_run / "map.csv")) == 4 * 5
    comp = _rows(quick_run / "comparisons.csv")
    assert {r["experiment"] for r in comp} == {
        "detection",
        "equal_plv",
        "equal_moduli",
        "trap",
    }
    assert {r["statistic"] for r in comp} >= {
        "plv",
        "plv2",
        "qmi_rich",
        "bures_rich",
        "qmi_weighted_02",
        "qmi_weighted_05",
    }
    meta = json.loads((quick_run / "run_metadata.json").read_text())
    assert (
        meta["config"]["protocol"]["n_records"] == 6
        and meta["comparisons_not_made"] == []
    )
    cont = json.loads((quick_run / "continuity.json").read_text())
    assert cont["spearman_abs_m1_abs_m2"] == 1.0
    assert cont["ranks_identical_abs_m1_abs_m2"] is True
    assert cont["ranks_identical_abs_m1_qmi_rich"] is True


def test_an_unreachable_equal_moduli_pair_is_reported_not_hidden(tmp_path):
    cfg = tmp_path / "c.yaml"
    base = (
        __import__("pathlib").Path(__file__).parents[1] / "config" / "base.yaml"
    ).read_text()
    cfg.write_text(base.replace("k2_ref: 0.12", "k2_ref: 0.15"))
    assert main(["--quick", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 0
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["comparisons_not_made"][0]["experiment"] == "equal_moduli"
    assert "closest" in meta["comparisons_not_made"][0]["reason"]
    assert "equal_moduli" not in {
        r["experiment"] for r in _rows(tmp_path / "out" / "comparisons.csv")
    }


def test_standard_error_of_the_modulus_projects_on_the_mean_direction():
    import numpy as np
    import pytest

    from scripts.run_second_harmonic_bench import mean_modulus_and_se

    # Scatter only across the mean direction: the modulus does not move to first order.
    across = 0.5 + 1j * np.array([-1.0, 1.0, -0.3, 0.3])
    assert mean_modulus_and_se(across) == (0.5, 0.0)
    # Scatter only along it: the usual standard error of a real mean.
    deviations = np.array([-0.2, 0.1, 0.05, 0.05, -0.1, 0.1])
    modulus, se = mean_modulus_and_se((0.5 + deviations) * np.exp(0.7j))
    assert modulus == pytest.approx(0.5)
    assert se == pytest.approx(np.std(deviations, ddof=1) / np.sqrt(deviations.size))
    # A mean of exactly zero has no direction: the standard error is undefined.
    modulus, se = mean_modulus_and_se(np.array([1.0 + 0j, -1.0 + 0j]))
    assert modulus == 0.0 and np.isnan(se)
    with pytest.raises(ValueError):
        mean_modulus_and_se(np.array([1.0 + 0j]))


def test_a_timestep_check_with_one_record_is_rejected_by_name(tmp_path):
    import pytest

    cfg = tmp_path / "c.yaml"
    base = (
        __import__("pathlib").Path(__file__).parents[1] / "config" / "base.yaml"
    ).read_text()
    cfg.write_text(
        base.replace(
            "timestep_check: {duration_s: 40.0, n_records: 4}",
            "timestep_check: {duration_s: 40.0, n_records: 1}",
        )
    )
    with pytest.raises(ValueError, match="timestep_check.n_records"):
        main(["--quick", "--config", str(cfg), "--out", str(tmp_path / "out")])


def test_a_missing_git_binary_records_the_commit_as_unknown(monkeypatch):
    import scripts.run_second_harmonic_bench as bench

    def _no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(bench.subprocess, "run", _no_git)
    assert bench._git_commit() == {"commit": None, "dirty": None}


def test_reliability_has_no_trap_row_while_comparisons_and_summary_keep_it(quick_run):
    assert "trap" not in {r["experiment"] for r in _rows(quick_run / "reliability.csv")}
    assert "trap" in {r["experiment"] for r in _rows(quick_run / "comparisons.csv")}
    assert "trap" in {r["experiment"] for r in _rows(quick_run / "summary.csv")}


def test_reliability_reports_the_record_counts_at_the_shortest_duration(quick_run):
    summary = _rows(quick_run / "summary.csv")
    reliability = _rows(quick_run / "reliability.csv")
    assert reliability[0].keys() >= {"mean_n_valid", "mean_n_null"}
    reported = 0
    for row in reliability:
        if row["shortest_reliable_duration_s"] == "":
            assert row["mean_n_valid"] == "" and row["mean_n_null"] == ""
            continue
        reported += 1
        match = [
            s
            for s in summary
            if (s["experiment"], s["label"], s["statistic"])
            == (row["experiment"], row["label"], row["statistic"])
            and float(s["duration_s"]) == float(row["shortest_reliable_duration_s"])
        ]
        assert len(match) == 1
        assert row["mean_n_valid"] == match[0]["mean_n_valid"]
        assert row["mean_n_null"] == match[0]["mean_n_null"]
    assert reported > 0


def test_summary_averages_the_null_standard_deviations_over_seeds(quick_run):
    summary = _rows(quick_run / "summary.csv")
    comparisons = _rows(quick_run / "comparisons.csv")
    assert list(summary[0])[-2:] == ["mean_null_sd_a", "mean_null_sd_b"]
    row = next(r for r in summary if r["mean_null_sd_a"] != "")
    group = [
        c
        for c in comparisons
        if (c["experiment"], c["label"], c["duration_s"], c["statistic"])
        == (row["experiment"], row["label"], row["duration_s"], row["statistic"])
    ]
    for side in ("a", "b"):
        values = [
            float(c[f"null_sd_{side}"]) for c in group if c[f"null_sd_{side}"] != ""
        ]
        expected = sum(values) / len(values) if values else None
        cell = row[f"mean_null_sd_{side}"]
        if expected is None:
            assert cell == ""
        else:
            assert float(cell) == pytest.approx(expected)


def _after(header, name, count):
    i = header.index(name)
    return header[i + 1 : i + 1 + count]


def test_comparisons_and_summary_carry_the_exact_values_of_the_scheme(quick_run):
    for name in ("comparisons.csv", "summary.csv"):
        with (quick_run / name).open() as fh:
            header = next(csv.reader(fh))
        assert _after(header, "exact_b", 2) == [
            "exact_discrete_a",
            "exact_discrete_b",
        ], name
    summary = _rows(quick_run / "summary.csv")
    trap = next(
        r for r in summary if r["experiment"] == "trap" and r["statistic"] == "plv2"
    )
    assert float(trap["exact_discrete_b"]) == pytest.approx(
        0.796789, abs=1e-6
    )  # 0.8018 in continuous time
    assert float(trap["exact_b"]) - float(trap["exact_discrete_b"]) > 4e-3


def test_matched_pairs_are_solved_on_the_simulated_scheme(quick_run):
    from qot_course.quantum_ot.biharmonic import discretised_stationary_moments

    meta = json.loads((quick_run / "run_metadata.json").read_text())
    dt = meta["config"]["model"]["dt"]
    solved = {s["experiment"]: s for s in reversed(meta["solved_settings"])}
    for entry in meta["solved_settings"]:
        if entry["experiment"] in ("equal_plv", "equal_moduli"):
            assert (
                entry["matched_on"] == "discretised scheme"
                and entry["matched_dt"] == dt
            )
        else:
            assert "matched_on" not in entry and "matched_dt" not in entry
    plv = solved["equal_plv"]
    for side in ("condition_a", "condition_b"):
        m = discretised_stationary_moments(*plv[side], dt=dt)
        assert abs(m[1]) == pytest.approx(plv["target"], abs=1e-6)
    pair = solved["equal_moduli"]["pair"]
    ref = discretised_stationary_moments(pair["k1_ref"], pair["k2_ref"], dt=dt)
    other = discretised_stationary_moments(pair["k1"], pair["k2"], dt=dt)
    residual = max(abs(abs(other[h]) - abs(ref[h])) for h in (1, 2))
    assert pair["residual"] == pytest.approx(residual, abs=1e-12)


def test_timestep_check_reports_the_gap_to_the_scheme(quick_run):
    from qot_course.quantum_ot.biharmonic import discretised_stationary_moments

    with (quick_run / "timestep_check.csv").open() as fh:
        header = next(csv.reader(fh))
    assert header == [
        "k1",
        "k2",
        "dt",
        "n_records",
        "abs_m1_exact",
        "abs_m1_sim",
        "abs_m1_se",
        "abs_m1_gap_se",
        "abs_m1_exact_discrete",
        "abs_m1_gap_discrete_se",
        "abs_m2_exact",
        "abs_m2_sim",
        "abs_m2_se",
        "abs_m2_gap_se",
        "abs_m2_exact_discrete",
        "abs_m2_gap_discrete_se",
    ]
    for row in _rows(quick_run / "timestep_check.csv"):
        m = discretised_stationary_moments(
            float(row["k1"]), float(row["k2"]), dt=float(row["dt"])
        )
        for h in (1, 2):
            assert float(row[f"abs_m{h}_exact_discrete"]) == pytest.approx(
                abs(m[h]), abs=1e-12
            )
            if row[f"abs_m{h}_se"] not in ("", "0.0"):
                gap = (float(row[f"abs_m{h}_sim"]) - abs(m[h])) / float(
                    row[f"abs_m{h}_se"]
                )
                assert float(row[f"abs_m{h}_gap_discrete_se"]) == pytest.approx(
                    gap, rel=1e-9
                )
