import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ive

from qot_course.quantum_ot.biharmonic import (
    relative_phase,
    simulate_biharmonic_dyad,
    stationary_moments,
)
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
     ({"noise_std": -0.1}, "noise_std"), ({"noise_std": 0.0}, "noise_std"), ({"burn_in": -1.0}, "burn_in"),
     ({"duration": 0.05}, "duration")],   # fewer than two samples
)
def test_invalid_arguments_name_the_offender(kwargs, name):
    with pytest.raises(ValueError, match=name):
        simulate_biharmonic_dyad(0.1, 0.1, **kwargs)


def test_bessel_ratio_without_detuning_or_second_harmonic():
    kappa = 2 * 0.2 / 0.3**2
    m = stationary_moments(0.2, 0.0, omega_1=1.0, omega_2=1.0)
    assert m[0] == pytest.approx(1.0)
    assert m[1].real == pytest.approx(ive(1, kappa) / ive(0, kappa), abs=1e-10)
    assert m[2].real == pytest.approx(ive(2, kappa) / ive(0, kappa), abs=1e-10)
    assert abs(m[1].imag) < 1e-10


@pytest.mark.parametrize("k1, k2", [(0.1, 0.1), (0.06, -0.10), (0.0, 0.15)])
def test_generalized_von_mises_without_detuning(k1, k2):
    d = 0.3**2
    w = lambda x: np.exp((2 * k1 * np.cos(x) + k2 * np.cos(2 * x)) / d)  # noqa: E731
    z = quad(w, -np.pi, np.pi)[0]
    m = stationary_moments(k1, k2, omega_1=1.0, omega_2=1.0)
    for h in (1, 2):
        exact = quad(lambda x: w(x) * np.cos(h * x), -np.pi, np.pi)[0] / z
        assert m[h].real == pytest.approx(exact, abs=1e-8)


def test_pure_second_harmonic_hides_from_the_first_moment_even_with_detuning():
    m = stationary_moments(0.0, 0.10)            # default detuning 0.2 rad/s
    assert abs(m[1]) < 1e-10
    assert abs(m[2]) == pytest.approx(0.3623, abs=5e-4)   # value found by the trial


def test_uncoupled_pair_has_no_moment():
    m = stationary_moments(0.0, 0.0)
    assert abs(m[1]) < 1e-12 and abs(m[2]) < 1e-12


def test_truncation_has_converged_in_the_bench_range():
    for k1, k2 in [(0.5, 0.5), (0.5, -0.5), (0.3, 0.0)]:
        np.testing.assert_allclose(
            stationary_moments(k1, k2, n_modes=200), stationary_moments(k1, k2, n_modes=400),
            atol=1e-10,
        )


def test_unconverged_truncation_raises_and_names_n_modes():
    with pytest.raises(ValueError, match="n_modes"):
        stationary_moments(4.0, 0.0, n_modes=8)


def test_agrees_with_a_long_simulation():
    m = stationary_moments(0.1, 0.1)
    t1, t2 = simulate_biharmonic_dyad(0.1, 0.1, n_records=64, duration=2000.0, seed=1)
    per_record = np.exp(1j * (t1 - t2)).mean(axis=1)
    se = per_record.std(ddof=1) / np.sqrt(64)
    # Five standard errors plus a 0.01 allowance for the Euler-Maruyama step, which Task 7 measures.
    assert abs(per_record.mean() - m[1]) < 5 * se + 0.01


def test_relative_phase_sign_follows_k2():
    assert relative_phase(stationary_moments(0.10, 0.05)) == pytest.approx(0.363, abs=5e-3)
    assert relative_phase(stationary_moments(0.10, -0.05)) == pytest.approx(-0.329, abs=5e-3)
    assert np.isnan(relative_phase(stationary_moments(0.0, 0.1)))


def test_zero_noise_is_rejected():
    with pytest.raises(ValueError, match="noise_std"):
        stationary_moments(0.1, 0.1, noise_std=0.0)


@pytest.mark.parametrize(
    "kwargs, name",
    [({"noise_std": -0.1}, "noise_std"), ({"n_modes": 0}, "n_modes"), ({"n_modes": 3}, "n_modes"),
     ({"max_harmonic": -1}, "max_harmonic")],
)
def test_invalid_moment_arguments_name_the_offender(kwargs, name):
    with pytest.raises(ValueError, match=name):
        stationary_moments(0.1, 0.1, **kwargs)


def test_relative_phase_needs_two_harmonics():
    with pytest.raises(ValueError, match="moments"):
        relative_phase(np.array([1.0 + 0j, 0.5 + 0j]))
