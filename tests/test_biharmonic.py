import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ive

from qot_course.quantum_ot.biharmonic import (
    EqualModuliPair,
    discretised_stationary_moments,
    long_run_joint_density,
    relative_phase,
    simulate_biharmonic_dyad,
    solve_equal_moduli_pair,
    solve_k1_for_plv,
    stationary_moments,
)
from qot_course.quantum_ot.capstone import coupling_qmi, simulate_kuramoto_dyad
from qot_course.quantum_ot.embeddings import (
    joint_density_from_states,
    weighted_multifreq_state,
)


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


def test_relative_phase_at_opposite_second_harmonics():
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


def _synthetic_pair(n=200_000, seed=0):
    """Phases whose sum is spread independently of their difference."""
    rng = np.random.default_rng(seed)
    delta = rng.vonmises(0.4, 1.5, n) + 0.3 * np.sin(2 * rng.uniform(0, 2 * np.pi, n))
    total = rng.uniform(0, 4 * np.pi, n)
    return (total + delta) / 2, (total - delta) / 2


@pytest.mark.parametrize("weights", [None, (1.0, 0.2, 2.0)])
def test_matches_the_empirical_density_of_a_long_record(weights):
    ta, tb = _synthetic_pair()
    d = ta - tb
    m = np.array([1.0, np.exp(1j * d).mean(), np.exp(2j * d).mean()])
    emp = joint_density_from_states(
        weighted_multifreq_state(ta, weights=weights), weighted_multifreq_state(tb, weights=weights)
    )
    model = long_run_joint_density(m, weights=weights)
    surviving = np.abs(model) > 0
    np.testing.assert_allclose(emp[surviving], model[surviving], atol=1e-10)   # exact up to rounding
    assert np.abs(emp[~surviving]).max() < 0.01                                # sampling noise only


def test_is_a_density_and_gives_zero_information_when_uncoupled():
    rho = long_run_joint_density(np.array([1.0, 0.0, 0.0]))
    assert rho.shape == (9, 9)
    assert np.trace(rho).real == pytest.approx(1.0)
    np.testing.assert_allclose(rho, rho.conj().T)
    assert coupling_qmi(rho, dims=(3, 3)) == pytest.approx(0.0, abs=1e-10)


def test_qubit_case_carries_the_first_moment_only():
    rho = long_run_joint_density(np.array([1.0, 0.4 + 0.1j]), harmonics=(1,))
    assert rho.shape == (4, 4)
    assert rho[2, 1] == pytest.approx((0.4 + 0.1j) / 4)   # |1>_A|0>_B against |0>_A|1>_B


def test_information_depends_on_the_relative_phase():
    q = [coupling_qmi(long_run_joint_density(np.array([1.0, 0.3, 0.3 * np.exp(1j * b)])), dims=(3, 3))
         for b in (0.0, np.pi)]
    assert q[0] == pytest.approx(0.0484, abs=2e-4)   # values found on 2026-11-11
    assert q[1] == pytest.approx(0.0554, abs=2e-4)


def test_too_few_moments_is_rejected():
    with pytest.raises(ValueError, match="moments"):
        long_run_joint_density(np.array([1.0, 0.3]))


@pytest.mark.parametrize("target, k2", [(0.3, 0.0), (0.3, -0.10), (0.6, -0.15), (0.6, 0.04)])
def test_matched_phase_locking_value(target, k2):
    k1 = solve_k1_for_plv(target, k2)
    assert abs(stationary_moments(k1, k2)[1]) == pytest.approx(target, abs=1e-6)


def test_matched_settings_differ_in_the_second_moment():
    a = stationary_moments(solve_k1_for_plv(0.3, 0.0), 0.0)
    b = stationary_moments(solve_k1_for_plv(0.3, -0.10), -0.10)
    assert abs(a[2]) == pytest.approx(0.0712, abs=5e-4)   # values found by the trial
    assert abs(b[2]) == pytest.approx(0.3681, abs=5e-4)


def test_unreachable_target_raises_and_names_the_setting():
    with pytest.raises(ValueError, match=r"target=0\.999.*k2=0\.1"):
        solve_k1_for_plv(0.999, 0.1, k1_search_max=0.2)


def test_equal_moduli_pair_is_found_within_the_bound():
    pair = solve_equal_moduli_pair(0.05, 0.12)
    assert isinstance(pair, EqualModuliPair) and pair.found
    assert pair.k2 < 0 and abs(pair.k2) <= 0.15
    ref, other = stationary_moments(0.05, 0.12), stationary_moments(pair.k1, pair.k2)
    np.testing.assert_allclose(np.abs(other[1:]), np.abs(ref[1:]), atol=1e-6)
    assert relative_phase(ref) - relative_phase(other) > 0.5      # about +0.18 against -0.46


def test_equal_moduli_pair_out_of_reach_is_flagged_not_raised():
    pair = solve_equal_moduli_pair(0.05, 0.15)       # needs |k2| > 0.15
    assert not pair.found and pair.residual > 1e-3
    assert abs(pair.k2) <= 0.15


@pytest.mark.parametrize(
    "solve",
    [
        lambda **kw: solve_k1_for_plv(0.3, 0.0, **kw),
        lambda **kw: solve_equal_moduli_pair(0.05, 0.12, **kw),
    ],
    ids=["plv", "equal_moduli"],
)
@pytest.mark.parametrize(
    "keyword, value", [("max_harmonic", 1), ("omega", 1.0)], ids=["max_harmonic", "unknown"]
)
def test_matched_solvers_reject_keywords_they_do_not_forward(solve, keyword, value):
    # With max_harmonic=1 the pair solver would match |m1| alone and report a false pair.
    with pytest.raises(ValueError, match=keyword):
        solve(**{keyword: value})


# ---------------------------------------------------------------------------
# Exact stationary moments of the Euler-Maruyama scheme
# ---------------------------------------------------------------------------

BENCH_SETTINGS = [(0.3, 0.0), (0.1, 0.1), (0.0, 0.1), (0.06, -0.10)]


@pytest.mark.parametrize("k1, k2", [(0.3, 0.0), (0.1, 0.1), (0.0, 0.1), (0.06, -0.10), (0.0, 0.3)])
def test_discretised_moments_tend_to_the_continuous_ones_at_a_small_step(k1, k2):
    # The step bias is first order in dt. At dt = 0.05 the largest component gap to the
    # continuous-time moments over these settings is 5.5e-3 (trap setting), so at dt = 1e-4,
    # 500 times smaller, the expected gap is at most about 1.1e-5. The tolerance 2e-5 allows
    # for that bias and nothing else: the numerical error of either solver is below 1e-12.
    discrete = discretised_stationary_moments(k1, k2, dt=1e-4)
    continuous = stationary_moments(k1, k2)
    np.testing.assert_allclose(discrete, continuous, rtol=0.0, atol=2e-5)


def test_discretised_moments_converge_at_first_order_in_the_step():
    continuous = stationary_moments(0.3, 0.0)
    gaps = [abs(discretised_stationary_moments(0.3, 0.0, dt=dt)[2] - continuous[2]) for dt in (0.05, 0.025)]
    assert gaps[1] > 1e-4                       # far above rounding
    assert gaps[0] / gaps[1] == pytest.approx(2.0, abs=0.1)


def _modulus_and_se(z):
    """Modulus of the mean and its standard error by projection on the mean direction."""
    mean = z.mean()
    projected = (z * np.conj(mean / abs(mean))).real
    return abs(mean), projected.std(ddof=1) / np.sqrt(z.size)


def test_discretised_moments_describe_the_simulated_scheme_not_the_model():
    # At dt = 0.5 the step bias of |m2| is about -0.04, far above the standard error of
    # 256 records of 2,000 s. The circle chain has a stationary law at any step, and with
    # K1 = 0.3 the locked state contracts by a factor of about 0.7 per step.
    k1, k2, dt = 0.3, 0.0, 0.5
    t1, t2 = simulate_biharmonic_dyad(k1, k2, n_records=256, duration=2000.0, dt=dt, burn_in=100.0, seed=4)
    delta = t1 - t2
    discrete = discretised_stationary_moments(k1, k2, dt=dt)
    continuous = stationary_moments(k1, k2)
    for h in (1, 2):
        modulus, se = _modulus_and_se(np.exp(1j * h * delta).mean(axis=1))
        assert abs(modulus - abs(discrete[h])) < 4 * se, h
        assert abs(modulus - abs(continuous[h])) > 5 * se, h


@pytest.mark.parametrize("k1, k2", BENCH_SETTINGS + [(0.0, 0.3)])
def test_discretised_truncation_has_converged_in_the_bench_range(k1, k2):
    np.testing.assert_allclose(
        discretised_stationary_moments(k1, k2, dt=0.05, n_modes=200),
        discretised_stationary_moments(k1, k2, dt=0.05, n_modes=400),
        rtol=0.0, atol=1e-10,
    )


@pytest.mark.parametrize(
    "setting, expected_gap", zip(BENCH_SETTINGS, (-0.00356, -0.00382, -0.00316, -0.00349)),
)
def test_step_bias_of_the_second_moment_at_the_bench_step(setting, expected_gap):
    # Gaps predicted by the final review of the bench, rounded to five decimals.
    gap = abs(discretised_stationary_moments(*setting, dt=0.05)[2]) - abs(stationary_moments(*setting)[2])
    assert gap == pytest.approx(expected_gap, abs=2e-5)


def test_unconverged_discretised_truncation_raises_and_names_n_modes():
    with pytest.raises(ValueError, match="n_modes"):
        discretised_stationary_moments(0.0, 0.3, dt=1e-4, n_modes=8)


@pytest.mark.parametrize(
    "kwargs, name",
    [({"dt": 0.0}, "dt"), ({"dt": -0.05}, "dt"), ({"dt": np.inf}, "dt"), ({"dt": np.nan}, "dt"),
     ({"noise_std": 0.0}, "noise_std"), ({"noise_std": np.nan}, "noise_std"),
     ({"omega_1": np.inf}, "omega_1"), ({"omega_2": np.nan}, "omega_2"),
     ({"k1": np.nan}, "k1"), ({"k2": np.inf}, "k2"),
     ({"max_harmonic": -1}, "max_harmonic"), ({"n_modes": 3}, "n_modes")],
)
def test_invalid_discretised_arguments_name_the_offender(kwargs, name):
    args = {"k1": 0.1, "k2": 0.1, "dt": 0.05}
    args.update(kwargs)
    k1, k2 = args.pop("k1"), args.pop("k2")
    with pytest.raises(ValueError, match=name):
        discretised_stationary_moments(k1, k2, **args)


def test_discretised_moments_keep_the_layout_of_the_continuous_ones():
    m = discretised_stationary_moments(0.1, 0.1, dt=0.05, max_harmonic=4)
    assert m.shape == (5,) and m.dtype == np.complex128
    assert m[0] == 1.0


# ---------------------------------------------------------------------------
# Solvers matched on the discretised scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target, k2", [(0.3, 0.0), (0.3, -0.10), (0.6, 0.04)])
def test_matched_phase_locking_value_on_the_discretised_scheme(target, k2):
    k1 = solve_k1_for_plv(target, k2, dt=0.05)
    assert abs(discretised_stationary_moments(k1, k2, dt=0.05)[1]) == pytest.approx(target, abs=1e-6)
    k1_continuous = solve_k1_for_plv(target, k2)
    assert 1e-5 < abs(k1 - k1_continuous) < 0.02


def test_equal_moduli_pair_on_the_discretised_scheme():
    pair = solve_equal_moduli_pair(0.05, 0.12, dt=0.05)
    assert pair.found and pair.k2 < 0
    ref = discretised_stationary_moments(0.05, 0.12, dt=0.05)
    other = discretised_stationary_moments(pair.k1, pair.k2, dt=0.05)
    np.testing.assert_allclose(np.abs(other[1:]), np.abs(ref[1:]), atol=1e-6)
    continuous = solve_equal_moduli_pair(0.05, 0.12)
    shift = np.hypot(pair.k1 - continuous.k1, pair.k2 - continuous.k2)
    assert 1e-5 < shift < 0.02


@pytest.mark.parametrize(
    "solve",
    [lambda: solve_k1_for_plv(0.3, 0.0, dt=0.0), lambda: solve_equal_moduli_pair(0.05, 0.12, dt=-0.1)],
    ids=["plv", "equal_moduli"],
)
def test_matched_solvers_reject_a_non_positive_step(solve):
    with pytest.raises(ValueError, match="dt"):
        solve()
