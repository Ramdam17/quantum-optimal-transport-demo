import numpy as np
import pytest

from qot_course.geometry.info_geometry import (
    bernoulli_fisher,
    fisher_rao_distance,
    fisher_rao_geodesic,
    mixture_interpolation,
    multinomial_fisher,
    simplex_to_cartesian,
    wasserstein_interpolation_1d,
)


# ----------------------------- Fisher information -------------------------- #
def test_bernoulli_fisher_minimum_at_one_half_and_diverges_at_boundary():
    assert bernoulli_fisher(0.5) == pytest.approx(4.0)
    assert bernoulli_fisher(0.01) > 100.0
    assert bernoulli_fisher(0.99) > 100.0


def test_bernoulli_fisher_rejects_boundary():
    with pytest.raises(ValueError):
        bernoulli_fisher(0.0)
    with pytest.raises(ValueError):
        bernoulli_fisher(1.0)


def test_multinomial_fisher_is_inverse_diagonal():
    p = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(multinomial_fisher(p), np.diag(1.0 / p))


def test_multinomial_fisher_rejects_zero_probabilities():
    with pytest.raises(ValueError):
        multinomial_fisher(np.array([0.5, 0.5, 0.0]))


# ----------------------------- Fisher–Rao distance ------------------------- #
def test_fisher_rao_distance_basic_properties():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.6, 0.3])
    assert fisher_rao_distance(p, p) == pytest.approx(0.0, abs=1e-10)
    assert fisher_rao_distance(p, q) > 0.0
    assert fisher_rao_distance(p, q) == pytest.approx(fisher_rao_distance(q, p))


def test_fisher_rao_distance_orthogonal_corners_equals_pi():
    # Disjoint supports => sqrt(p_i q_i) = 0 for all i => arccos(0) = pi/2 => d = pi.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    assert fisher_rao_distance(e1, e2) == pytest.approx(np.pi)


# ----------------------------- Fisher–Rao geodesic ------------------------- #
def test_fisher_rao_geodesic_endpoints():
    p = np.array([0.3, 0.4, 0.3])
    q = np.array([0.5, 0.1, 0.4])
    np.testing.assert_allclose(fisher_rao_geodesic(p, q, 0.0), p, atol=1e-12)
    np.testing.assert_allclose(fisher_rao_geodesic(p, q, 1.0), q, atol=1e-12)


def test_fisher_rao_geodesic_stays_on_simplex():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.1, 0.4, 0.5])
    for t in np.linspace(0.0, 1.0, 11):
        gt = fisher_rao_geodesic(p, q, t)
        assert np.all(gt >= -1e-12)
        assert gt.sum() == pytest.approx(1.0)


def test_fisher_rao_geodesic_length_matches_distance():
    # Sum of segment lengths of a discretized geodesic must approach the closed-form
    # Fisher--Rao distance as the partition is refined.
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    ts = np.linspace(0.0, 1.0, 200)
    arc = np.array([fisher_rao_geodesic(p, q, t) for t in ts])
    seg_len = np.sum(
        [fisher_rao_distance(arc[k], arc[k + 1]) for k in range(len(ts) - 1)]
    )
    assert seg_len == pytest.approx(fisher_rao_distance(p, q), rel=1e-3)


# ----------------------------- Interpolations ------------------------------ #
def test_mixture_interpolation_endpoints_and_midpoint():
    p = np.array([0.1, 0.5, 0.4])
    q = np.array([0.4, 0.4, 0.2])
    np.testing.assert_allclose(mixture_interpolation(p, q, 0.0), p)
    np.testing.assert_allclose(mixture_interpolation(p, q, 1.0), q)
    np.testing.assert_allclose(mixture_interpolation(p, q, 0.5), 0.5 * (p + q))


def _normalized_bump(support: np.ndarray, center: float, sigma: float) -> np.ndarray:
    raw = np.exp(-0.5 * ((support - center) / sigma) ** 2)
    return raw / raw.sum()


def test_wasserstein_interpolation_endpoints_are_close_to_inputs():
    support = np.linspace(0.0, 10.0, 100)
    p = _normalized_bump(support, 3.0, 1.0)
    q = _normalized_bump(support, 7.0, 1.0)
    np.testing.assert_allclose(
        wasserstein_interpolation_1d(p, q, support, 0.0), p, atol=5e-3
    )
    np.testing.assert_allclose(
        wasserstein_interpolation_1d(p, q, support, 1.0), q, atol=5e-3
    )


def test_mixture_is_bimodal_but_wasserstein_slides_for_separated_bumps():
    """The S6 punchline: same endpoints, two geometries, two midpoints."""
    support = np.linspace(0.0, 24.0, 200)
    p0 = _normalized_bump(support, 4.0, 1.2)
    p1 = _normalized_bump(support, 20.0, 1.2)

    pt_mixture = mixture_interpolation(p0, p1, 0.5)
    pt_w2 = wasserstein_interpolation_1d(p0, p1, support, 0.5)

    # Mixture midpoint: bimodal at the original centres, ~no mass in the gap.
    mass_ends_mix = pt_mixture[support < 8].sum() + pt_mixture[support > 16].sum()
    mass_mid_mix = pt_mixture[(support >= 8) & (support <= 16)].sum()
    assert mass_ends_mix > 0.95
    assert mass_mid_mix < 0.05

    # Wasserstein midpoint: ~all mass near x = 12 (single peak slid to the middle).
    mass_mid_w2 = pt_w2[(support >= 8) & (support <= 16)].sum()
    assert mass_mid_w2 > 0.95


# ----------------------------- Cartesian mapping --------------------------- #
def test_simplex_to_cartesian_vertices_and_centroid():
    np.testing.assert_allclose(simplex_to_cartesian([1.0, 0.0, 0.0]), [0.0, 0.0])
    np.testing.assert_allclose(simplex_to_cartesian([0.0, 1.0, 0.0]), [1.0, 0.0])
    np.testing.assert_allclose(
        simplex_to_cartesian([0.0, 0.0, 1.0]), [0.5, np.sqrt(3.0) / 2.0]
    )
    # Centroid of the equilateral triangle.
    np.testing.assert_allclose(
        simplex_to_cartesian([1 / 3, 1 / 3, 1 / 3]),
        [0.5, np.sqrt(3.0) / 6.0],
        atol=1e-12,
    )
