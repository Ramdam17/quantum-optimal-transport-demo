import numpy as np
import pytest

from qot_course.quantum_ot.biharmonic import simulate_biharmonic_dyad
from qot_course.quantum_ot.capstone import simulate_kuramoto_dyad


def test_shapes_and_reproducibility():
    a1, a2 = simulate_biharmonic_dyad(0.1, 0.05, n_records=3, duration=10.0, seed=7)
    b1, b2 = simulate_biharmonic_dyad(0.1, 0.05, n_records=3, duration=10.0, seed=7)
    assert a1.shape == a2.shape == (3, 200)
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)


def test_reduces_to_the_course_dyad_when_k2_is_zero():
    # Same start (both phases at zero), no discard, one record: the course's simulator.
    t1, t2 = simulate_kuramoto_dyad(K=0.4, duration=20.0, dt=0.05, seed=3)
    s1, s2 = simulate_biharmonic_dyad(
        0.4, 0.0, n_records=1, duration=20.0, dt=0.05,
        burn_in=0.0, random_initial_phases=False, seed=3,
    )
    np.testing.assert_allclose(s1[0], t1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(s2[0], t2, rtol=0.0, atol=1e-12)


def test_records_are_independent_and_start_spread_out():
    t1, _ = simulate_biharmonic_dyad(0.0, 0.0, n_records=400, duration=1.0, seed=0)
    start = np.exp(1j * t1[:, 0])
    assert abs(start.mean()) < 0.15          # uniform start, not all at zero
    assert np.unique(t1[:, 0]).size == 400


def test_seed_sequence_is_accepted():
    ss = np.random.SeedSequence(11)
    a1, _ = simulate_biharmonic_dyad(0.1, 0.1, n_records=2, duration=2.0, seed=ss)
    assert a1.shape == (2, 40)


@pytest.mark.parametrize(
    "kwargs, name",
    [({"dt": 0.0}, "dt"), ({"duration": -1.0}, "duration"), ({"n_records": 0}, "n_records"),
     ({"noise_std": -0.1}, "noise_std"), ({"burn_in": -1.0}, "burn_in"),
     ({"duration": 0.05}, "duration")],   # fewer than two samples
)
def test_invalid_arguments_name_the_offender(kwargs, name):
    with pytest.raises(ValueError, match=name):
        simulate_biharmonic_dyad(0.1, 0.1, **kwargs)
