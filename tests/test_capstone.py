import numpy as np
import pytest

from qot_course.quantum.composite import partial_trace
from qot_course.quantum_ot.capstone import (
    cosine_correlation,
    coupling_bures,
    coupling_qmi,
    joint_density_matrix,
    plv,
    simulate_kuramoto_dyad,
)


# ----------------------------- Simulator -------------------------------- #
def test_simulator_returns_matched_length_trajectories():
    t1, t2 = simulate_kuramoto_dyad(K=0.5, duration=10.0, dt=0.1, seed=0)
    expected = int(10.0 / 0.1)
    assert t1.shape == (expected,)
    assert t2.shape == (expected,)


def test_simulator_is_reproducible_with_same_seed():
    t1a, t2a = simulate_kuramoto_dyad(K=0.5, duration=5.0, dt=0.05, seed=42)
    t1b, t2b = simulate_kuramoto_dyad(K=0.5, duration=5.0, dt=0.05, seed=42)
    np.testing.assert_array_equal(t1a, t1b)
    np.testing.assert_array_equal(t2a, t2b)


# ----------------------------- Density matrix --------------------------- #
def test_joint_density_matrix_is_a_valid_state():
    t1, t2 = simulate_kuramoto_dyad(K=1.0, duration=200.0, seed=0)
    rho = joint_density_matrix(t1, t2)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)  # Hermitian
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-9)  # trace 1
    eigs = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    assert np.min(eigs) > -1e-9  # PSD


def test_uncoupled_dyad_gives_marginals_near_maximally_mixed():
    """K=0 with long enough simulation: phases drift uniformly and marginals → I/2."""
    t1, t2 = simulate_kuramoto_dyad(K=0.0, duration=500.0, seed=1)
    rho = joint_density_matrix(t1, t2)
    rho_a = partial_trace(rho, keep=[0], dims=[2, 2])
    rho_b = partial_trace(rho, keep=[1], dims=[2, 2])
    np.testing.assert_allclose(rho_a, np.eye(2) / 2, atol=0.05)
    np.testing.assert_allclose(rho_b, np.eye(2) / 2, atol=0.05)


# ----------------------------- Coupling measures track K --------------- #
def _measures_at_k(K: float, duration: float = 200.0):
    t1, t2 = simulate_kuramoto_dyad(K=K, duration=duration, seed=0)
    rho = joint_density_matrix(t1, t2)
    return {
        "qmi": coupling_qmi(rho),
        "bures": coupling_bures(rho),
        "plv": plv(t1, t2),
        "corr": abs(cosine_correlation(t1, t2)),
    }


def test_coupling_measures_at_zero_K_are_small():
    m = _measures_at_k(0.0, duration=500.0)
    # All measures should be near zero for uncoupled oscillators (with enough samples).
    assert m["qmi"] < 0.05
    assert m["bures"] < 0.3
    assert m["plv"] < 0.2
    assert m["corr"] < 0.2


def test_coupling_measures_increase_with_K():
    """All measures should be larger at K=4 than at K=0.5."""
    m_low = _measures_at_k(0.5)
    m_high = _measures_at_k(4.0)
    assert m_high["qmi"] > m_low["qmi"]
    assert m_high["bures"] > m_low["bures"]
    assert m_high["plv"] > m_low["plv"]


def test_qmi_is_nonnegative():
    for K in [0.0, 1.0, 2.0]:
        t1, t2 = simulate_kuramoto_dyad(K=K, duration=100.0, seed=0)
        rho = joint_density_matrix(t1, t2)
        assert coupling_qmi(rho) >= -1e-9


def test_bures_coupling_is_nonnegative_and_zero_for_uncoupled():
    """Bures coupling for K=0 should be small (since rho_AB ≈ I/4 = product)."""
    t1, t2 = simulate_kuramoto_dyad(K=0.0, duration=500.0, seed=2)
    rho = joint_density_matrix(t1, t2)
    assert coupling_bures(rho) < 0.3


# ----------------------------- PLV / classical correlation -------------- #
def test_plv_extremes():
    n = 5000
    rng = np.random.default_rng(0)
    # Uniformly random phases → PLV near 0.
    indep = rng.uniform(0, 2 * np.pi, size=n)
    indep2 = rng.uniform(0, 2 * np.pi, size=n)
    assert plv(indep, indep2) < 0.1
    # Identical phases → PLV = 1.
    same = rng.uniform(0, 2 * np.pi, size=n)
    assert plv(same, same) == pytest.approx(1.0)
