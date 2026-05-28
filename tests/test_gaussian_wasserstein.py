import numpy as np
import ot
import pytest

from qot_course.infotheory.quantum import bures_distance as quantum_bures_distance
from qot_course.quantum.density import density_matrix, maximally_mixed
from qot_course.quantum.states import KET_0, KET_PLUS
from qot_course.transport.gaussian import (
    bures_matrix_distance,
    bures_wasserstein_distance,
    gaussian_geodesic,
    gaussian_ot_map,
)


# ----------------------------- Bures matrix distance ----------------------- #
def test_bures_matrix_distance_zero_for_identical_matrices():
    Sigma = np.array([[1.5, 0.4], [0.4, 0.8]])
    assert bures_matrix_distance(Sigma, Sigma) == pytest.approx(0.0, abs=1e-9)


def test_bures_matrix_distance_diagonal_case_equals_sqrt_diff_squared():
    # For diagonal Sigma_0 = diag(a^2), Sigma_1 = diag(b^2), the formula reduces to
    # sum (a_i - b_i)^2 = || sqrt(Sigma_0) - sqrt(Sigma_1) ||_F^2.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 1.5, 4.0])
    Sigma_0 = np.diag(a**2)
    Sigma_1 = np.diag(b**2)
    expected = float(np.sum((a - b) ** 2))
    assert bures_matrix_distance(Sigma_0, Sigma_1) == pytest.approx(expected, abs=1e-9)


def test_bures_matrix_distance_matches_quantum_bures_squared_for_density_matrices():
    """The bridge: for unit-trace PSD matrices (density matrices), the Bures matrix
    distance equals the squared Bures distance from S7."""
    pairs = [
        (density_matrix(KET_PLUS), maximally_mixed(2)),
        (density_matrix(KET_0), density_matrix(KET_PLUS)),
        (density_matrix(KET_PLUS), density_matrix(KET_PLUS)),
    ]
    for rho, sigma in pairs:
        bures_sq_s7 = quantum_bures_distance(rho, sigma) ** 2
        bures_sq_s11 = bures_matrix_distance(rho, sigma)
        assert bures_sq_s11 == pytest.approx(bures_sq_s7, abs=1e-6)


# ----------------------------- Bures-Wasserstein on Gaussians ------------- #
def test_bures_wasserstein_translation_equals_mean_shift_norm():
    Sigma = np.array([[1.0, 0.3], [0.3, 1.2]])
    m_0 = np.array([0.0, 0.0])
    m_1 = np.array([2.5, -1.5])
    expected = float(np.linalg.norm(m_0 - m_1))
    assert bures_wasserstein_distance(m_0, Sigma, m_1, Sigma) == pytest.approx(expected)


def test_bures_wasserstein_1d_gaussian_closed_form():
    # 1-D Gaussian special case: W_2(N(m_0, s_0^2), N(m_1, s_1^2))
    # = sqrt( (m_0 - m_1)^2 + (s_0 - s_1)^2 ).
    m_0, m_1 = 0.0, 3.0
    s_0, s_1 = 1.0, 2.5
    expected = float(np.sqrt((m_0 - m_1) ** 2 + (s_0 - s_1) ** 2))
    w2 = bures_wasserstein_distance(
        [m_0], [[s_0**2]], [m_1], [[s_1**2]]
    )
    assert w2 == pytest.approx(expected, abs=1e-9)


def test_bures_wasserstein_matches_grid_lp_on_2d_gaussians():
    """Discretise both Gaussians on a 2-D grid, solve the LP, compare to closed form."""
    mean_0 = np.array([-2.0, -1.0])
    cov_0 = np.array([[1.0, 0.3], [0.3, 0.5]])
    mean_1 = np.array([2.0, 1.5])
    cov_1 = np.array([[0.5, -0.2], [-0.2, 1.0]])
    w2_cf = bures_wasserstein_distance(mean_0, cov_0, mean_1, cov_1)

    # Build a 20x20 grid covering both bumps with margin.
    grid = np.linspace(-6.0, 6.0, 20)
    X, Y = np.meshgrid(grid, grid)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (400, 2)

    def density(pts, mean, cov):
        diff = pts - mean
        inv = np.linalg.inv(cov)
        e = -0.5 * np.einsum("ij,jk,ik->i", diff, inv, diff)
        return np.exp(e)

    p0 = density(pts, mean_0, cov_0)
    p0 /= p0.sum()
    p1 = density(pts, mean_1, cov_1)
    p1 /= p1.sum()
    cost_grid = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    w2_lp = float(np.sqrt(ot.emd2(p0, p1, cost_grid)))
    # Grid discretisation has finite resolution; tolerance ~5% absolute.
    assert w2_cf == pytest.approx(w2_lp, rel=0.05)


# ----------------------------- Gaussian OT map ----------------------------- #
def test_gaussian_ot_map_pushforward_property():
    """A Sigma_0 A^T = Sigma_1 (the linear OT map pushes one cov onto the other)."""
    rng = np.random.default_rng(0)
    # Random SPD covariance matrices.
    L0 = rng.normal(size=(3, 3))
    cov_0 = L0 @ L0.T + 0.1 * np.eye(3)
    L1 = rng.normal(size=(3, 3))
    cov_1 = L1 @ L1.T + 0.1 * np.eye(3)
    A = gaussian_ot_map(cov_0, cov_1)
    pushed = A @ cov_0 @ A.T
    np.testing.assert_allclose(pushed, cov_1, atol=1e-9)


def test_gaussian_ot_map_is_symmetric_positive_definite():
    cov_0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    cov_1 = np.array([[1.0, -0.3], [-0.3, 3.0]])
    A = gaussian_ot_map(cov_0, cov_1)
    np.testing.assert_allclose(A, A.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(A) > 0.0)


# ----------------------------- Gaussian geodesic --------------------------- #
def test_gaussian_geodesic_endpoints():
    mean_0 = np.array([0.0, 0.0])
    cov_0 = np.array([[1.0, 0.2], [0.2, 0.5]])
    mean_1 = np.array([3.0, -1.0])
    cov_1 = np.array([[0.5, -0.1], [-0.1, 2.0]])
    m_t0, c_t0 = gaussian_geodesic(mean_0, cov_0, mean_1, cov_1, t=0.0)
    m_t1, c_t1 = gaussian_geodesic(mean_0, cov_0, mean_1, cov_1, t=1.0)
    np.testing.assert_allclose(m_t0, mean_0, atol=1e-9)
    np.testing.assert_allclose(c_t0, cov_0, atol=1e-9)
    np.testing.assert_allclose(m_t1, mean_1, atol=1e-9)
    np.testing.assert_allclose(c_t1, cov_1, atol=1e-9)


def test_gaussian_geodesic_covariance_stays_spd_along_path():
    mean_0 = np.array([0.0, 0.0])
    cov_0 = np.array([[1.5, 0.4], [0.4, 0.8]])
    mean_1 = np.array([1.0, 1.0])
    cov_1 = np.array([[0.6, -0.2], [-0.2, 1.6]])
    for t in np.linspace(0.0, 1.0, 11):
        _, cov_t = gaussian_geodesic(mean_0, cov_0, mean_1, cov_1, t)
        eigs = np.linalg.eigvalsh(0.5 * (cov_t + cov_t.T))
        assert np.all(eigs > -1e-9)
