import numpy as np
import pytest

from qot_course.quantum_ot.bench_protocol import (
    STATISTIC_KEYS_BASE, auc, compare_conditions, detection_rate, harmonic_plv,
    long_run_statistics, record_statistics, relative_phase_components, signed_score,
)
from qot_course.quantum_ot.biharmonic import simulate_biharmonic_dyad, stationary_moments
from qot_course.quantum_ot.capstone import plv

WEIGHTS = {"qmi_weighted_02": (1.0, 0.2, 2.0), "qmi_weighted_05": (1.0, 0.5, 2.0)}
MODEL = {"omega_1": 1.0, "omega_2": 1.2, "noise_std": 0.3, "dt": 0.05, "burn_in": 50.0, "n_modes": 200}


def test_first_harmonic_is_the_course_plv_and_second_sees_antiphase_mixtures():
    rng = np.random.default_rng(0)
    ta = rng.uniform(0, 2 * np.pi, 20_000)
    tb = ta - rng.choice([0.0, np.pi], 20_000)        # in phase or in antiphase, half each
    assert harmonic_plv(ta, tb, 1) == pytest.approx(plv(ta, tb))
    assert harmonic_plv(ta, tb, 1) < 0.03
    assert harmonic_plv(ta, tb, 2) == pytest.approx(1.0)


def test_relative_phase_components_are_invariant_under_a_common_shift_of_delta():
    m1, m2 = 0.4 * np.exp(0.7j), 0.3 * np.exp(1.9j)
    shifted = relative_phase_components(m1 * np.exp(1j * 0.5), m2 * np.exp(2j * 0.5))
    assert relative_phase_components(m1, m2) == pytest.approx(shifted)
    s, c = relative_phase_components(m1, m2)
    assert np.hypot(s, c) == pytest.approx(0.3)
    assert np.arctan2(s, c) == pytest.approx(1.9 - 1.4)


def test_relative_phase_components_are_undefined_near_zero_first_moment():
    assert all(np.isnan(relative_phase_components(0.01, 0.3)))


def test_record_statistics_keys_and_agreement_with_the_exact_values():
    t1, t2 = simulate_biharmonic_dyad(0.1, 0.1, n_records=1, duration=20_000.0, seed=2)
    rec = record_statistics(t1[0], t2[0], weight_sets=WEIGHTS)
    exact = long_run_statistics(stationary_moments(0.1, 0.1), weight_sets=WEIGHTS)
    assert tuple(rec) == STATISTIC_KEYS_BASE + tuple(WEIGHTS) == tuple(exact)
    for key in rec:
        assert rec[key] == pytest.approx(exact[key], abs=0.03), key


def test_uncoupled_exact_statistics_are_zero_and_relative_phase_undefined():
    exact = long_run_statistics(stationary_moments(0.0, 0.0), weight_sets=WEIGHTS)
    for key in ("plv", "plv2", "qmi_naive", "qmi_rich", "bures_rich", *WEIGHTS):
        assert exact[key] == pytest.approx(0.0, abs=1e-6), key
    assert np.isnan(exact["relphase_sin"]) and np.isnan(exact["relphase_cos"])


def test_score_keeps_its_sign_and_is_not_folded():
    null = np.array([-1.0, 1.0, -1.0, 1.0])                    # mean 0, sd(ddof=1) = 1.1547
    assert signed_score(np.full(4, -2.0), null) == pytest.approx(-2.0 / np.std(null, ddof=1))
    assert signed_score(np.array([2.0, -2.0]), null) == pytest.approx(0.0)


def test_score_is_nan_without_null_scatter_or_with_too_few_values():
    assert np.isnan(signed_score([1.0, 2.0], [3.0, 3.0, 3.0]))
    assert np.isnan(signed_score([1.0, np.nan], [np.nan, 0.5, np.nan]))


def test_auc_bounds_ties_and_invariance_under_a_monotone_map():
    a, b = np.array([0.1, 0.2, 0.3]), np.array([0.25, 0.4, 0.5])
    assert auc(a, a) == pytest.approx(0.5)
    assert auc(a, b + 1.0) == pytest.approx(1.0)
    assert auc(a, b) == pytest.approx(8 / 9)
    assert auc(a**2, b**2) == pytest.approx(auc(a, b))          # the score has no such property
    assert np.isnan(auc([np.nan], b))


def test_detection_rate_is_two_sided_and_calibrated():
    rng = np.random.default_rng(0)
    null = rng.standard_normal(20_000)
    assert 0.04 < detection_rate(rng.standard_normal(20_000), null) < 0.06
    assert detection_rate(null[:100] + 10.0, null) == pytest.approx(1.0)
    assert detection_rate(null[:100] - 10.0, null) == pytest.approx(1.0)   # lower tail counts


def test_compare_conditions_rows_and_reproducibility():
    kw = dict(duration=20.0, n_records=6, seed=5, model=MODEL, weight_sets={"qmi_weighted_02": (1.0, 0.2, 2.0)})
    rows = compare_conditions((0.0, 0.0), (0.0, 0.10), **kw)
    assert [r["statistic"] for r in rows] == [
        "plv", "plv2", "relphase_sin", "relphase_cos", "qmi_naive", "qmi_rich", "bures_rich", "qmi_weighted_02",
    ]
    # assert_equal walks lists and dictionaries and treats NaN as equal to NaN.
    np.testing.assert_equal(rows, compare_conditions((0.0, 0.0), (0.0, 0.10), **kw))
    by = {r["statistic"]: r for r in rows}
    assert by["plv2"]["exact_a"] == pytest.approx(0.0, abs=1e-9)
    assert by["plv2"]["exact_b"] == pytest.approx(0.3623, abs=5e-4)
    assert 0 <= by["relphase_sin"]["n_valid"] <= 6             # may be undefined: no exception
    assert by["plv"]["n_valid"] == 6


def test_compare_conditions_separates_a_clear_case():
    rows = compare_conditions(
        (0.0, 0.0), (0.0, 0.10), duration=800.0, n_records=40, seed=1, model=MODEL, weight_sets={},
    )
    by = {r["statistic"]: r for r in rows}
    assert by["plv2"]["auc"] > 0.95 and by["plv2"]["score"] > 3.0
    assert abs(by["plv"]["score"]) < 3.0                       # the first harmonic does not see it
