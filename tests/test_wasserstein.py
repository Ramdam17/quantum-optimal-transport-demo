import numpy as np
import ot
import pytest

from qot_course.transport.discrete import (
    cityblock_cost,
    squared_euclidean_cost,
)
from qot_course.transport.wasserstein import (
    mccann_geodesic_atoms_1d,
    wasserstein_1d,
    wasserstein_distance,
)


# ----------------------------- General W_p via LP -------------------------- #
def _normalised(x: np.ndarray) -> np.ndarray:
    return x / x.sum()


def test_wasserstein_2_matches_emd2_sqrt():
    rng = np.random.default_rng(0)
    a = _normalised(rng.random(5))
    b = _normalised(rng.random(7))
    positions_a = np.arange(5, dtype=float)
    positions_b = np.linspace(0.5, 4.5, 7)
    cost = squared_euclidean_cost(positions_a, positions_b)
    w2 = wasserstein_distance(a, b, cost, p=2)
    assert w2 == pytest.approx(np.sqrt(ot.emd2(a, b, cost)))


def test_wasserstein_1_matches_emd2_directly():
    rng = np.random.default_rng(1)
    a = _normalised(rng.random(4))
    b = _normalised(rng.random(4))
    positions = np.arange(4, dtype=float)
    cost = cityblock_cost(positions, positions)
    w1 = wasserstein_distance(a, b, cost, p=1)
    assert w1 == pytest.approx(ot.emd2(a, b, cost))


# ----------------------------- 1-D closed form ----------------------------- #
def test_wasserstein_1d_translated_dirac():
    # W_p between two Dirac masses at distance d is exactly d.
    for d in [0.0, 1.0, 3.5, 10.0]:
        for p in [1.0, 2.0, 3.0]:
            assert wasserstein_1d([0.0], [1.0], [d], [1.0], p=p) == pytest.approx(d)


def test_wasserstein_1d_translated_uniform():
    # A uniform distribution shifted by d has W_p = d for every p.
    positions = np.arange(10, dtype=float)
    weights = np.ones(10) / 10
    shift = 4.0
    for p in [1.0, 2.0, 3.5]:
        assert wasserstein_1d(positions, weights, positions + shift, weights, p=p) == (
            pytest.approx(shift)
        )


def test_wasserstein_1d_matches_lp_on_random_atomic_examples():
    rng = np.random.default_rng(2)
    for trial in range(8):
        n = rng.integers(2, 9)
        m = rng.integers(2, 9)
        pa = np.sort(rng.uniform(-3.0, 3.0, size=n))
        pb = np.sort(rng.uniform(-3.0, 3.0, size=m))
        wa = _normalised(rng.random(n))
        wb = _normalised(rng.random(m))
        cost = squared_euclidean_cost(pa, pb)
        lp_value = float(np.sqrt(ot.emd2(wa, wb, cost)))
        cf_value = wasserstein_1d(pa, wa, pb, wb, p=2)
        assert cf_value == pytest.approx(lp_value, abs=1e-9, rel=1e-7)


def test_wasserstein_1d_is_invariant_to_input_order():
    rng = np.random.default_rng(3)
    n = 6
    pa = rng.uniform(0, 5, size=n)
    pb = rng.uniform(0, 5, size=n)
    wa = _normalised(rng.random(n))
    wb = _normalised(rng.random(n))
    w_original = wasserstein_1d(pa, wa, pb, wb, p=2)
    perm = rng.permutation(n)
    w_shuffled = wasserstein_1d(pa[perm], wa[perm], pb, wb, p=2)
    assert w_shuffled == pytest.approx(w_original)


def test_wasserstein_1d_rejects_mass_mismatch():
    with pytest.raises(ValueError):
        wasserstein_1d([0.0], [1.0], [3.0], [0.5])


# ----------------------------- Metric axioms ------------------------------ #
def test_wasserstein_metric_axioms_on_1d_atomic_triple():
    """W_2 satisfies identity-of-indiscernibles, symmetry, and triangle inequality."""
    mu = ([0.0, 1.0, 2.0], [0.2, 0.3, 0.5])
    nu = ([0.5, 1.5, 2.5], [0.4, 0.4, 0.2])
    rho = ([-0.5, 0.0, 3.0], [0.1, 0.5, 0.4])
    d_mu_mu = wasserstein_1d(*mu, *mu, p=2)
    d_mu_nu = wasserstein_1d(*mu, *nu, p=2)
    d_nu_mu = wasserstein_1d(*nu, *mu, p=2)
    d_mu_rho = wasserstein_1d(*mu, *rho, p=2)
    d_nu_rho = wasserstein_1d(*nu, *rho, p=2)
    # Identity, symmetry, triangle.
    assert d_mu_mu == pytest.approx(0.0, abs=1e-12)
    assert d_mu_nu == pytest.approx(d_nu_mu)
    assert d_mu_rho <= d_mu_nu + d_nu_rho + 1e-9


# ----------------------------- McCann atomic geodesic --------------------- #
def test_mccann_geodesic_endpoints():
    pa, wa = np.array([0.0, 4.0]), np.array([0.5, 0.5])
    pb, wb = np.array([1.0, 2.0, 3.0]), np.array([1 / 3, 1 / 3, 1 / 3])
    p0, w0 = mccann_geodesic_atoms_1d(pa, wa, pb, wb, t=0.0)
    p1, w1 = mccann_geodesic_atoms_1d(pa, wa, pb, wb, t=1.0)
    # At t=0 the atom positions reproduce the source quantile values (with the
    # refined mass partition); at t=1 they reproduce the target.
    # Aggregate the atomic representation back into masses on the input grid.
    def gather(positions, weights, grid):
        out = np.zeros_like(grid, dtype=float)
        for x, w in zip(positions, weights):
            out[np.argmin(np.abs(grid - x))] += w
        return out

    np.testing.assert_allclose(gather(p0, w0, pa), wa, atol=1e-9)
    np.testing.assert_allclose(gather(p1, w1, pb), wb, atol=1e-9)


def test_mccann_midpoint_lies_between_source_and_target_positions():
    pa, wa = np.array([0.0]), np.array([1.0])
    pb, wb = np.array([10.0]), np.array([1.0])
    positions_half, _ = mccann_geodesic_atoms_1d(pa, wa, pb, wb, t=0.5)
    np.testing.assert_allclose(positions_half, [5.0])
