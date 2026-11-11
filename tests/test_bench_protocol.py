import numpy as np
import pytest

from qot_course.quantum_ot.bench_protocol import (
    STATISTIC_KEYS_BASE, harmonic_plv, long_run_statistics, record_statistics,
    relative_phase_components,
)
from qot_course.quantum_ot.biharmonic import simulate_biharmonic_dyad, stationary_moments
from qot_course.quantum_ot.capstone import plv

WEIGHTS = {"qmi_weighted_02": (1.0, 0.2, 2.0), "qmi_weighted_05": (1.0, 0.5, 2.0)}


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
