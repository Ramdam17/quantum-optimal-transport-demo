import numpy as np
import ot
import pytest

from qot_course.transport.discrete import (
    is_transportation_polytope_member,
    squared_euclidean_cost,
)
from qot_course.transport.sinkhorn import (
    gibbs_kernel,
    kl_to_kernel,
    sinkhorn,
    sinkhorn_distance,
    sinkhorn_iterations_log,
)


def _setup(n: int = 4, m: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    a = a / a.sum()
    b = rng.random(m)
    b = b / b.sum()
    cost = squared_euclidean_cost(np.arange(n, dtype=float), np.linspace(0, n - 1, m))
    return a, b, cost


# --------------------- Gibbs kernel + Sinkhorn marginals ------------------- #
def test_gibbs_kernel_basic_properties():
    cost = np.array([[0.0, 1.0], [2.0, 0.0]])
    K = gibbs_kernel(cost, epsilon=1.0)
    np.testing.assert_allclose(K, np.exp(-cost))
    # Larger epsilon -> kernel flatter (closer to ones); smaller -> sharper.
    K_large = gibbs_kernel(cost, epsilon=100.0)
    K_small = gibbs_kernel(cost, epsilon=0.1)
    assert K_large.min() > K_small.min()


def test_sinkhorn_plan_lies_in_transportation_polytope():
    a, b, cost = _setup()
    plan = sinkhorn(a, b, cost, epsilon=0.5)
    assert is_transportation_polytope_member(plan, a, b, atol=1e-6)


def test_sinkhorn_marginals_match_for_various_epsilons():
    # Standard (non-log-domain) Sinkhorn underflows when epsilon * max(C) is small
    # enough that exp(-C/epsilon) hits 0; we stay in the well-behaved regime here.
    a, b, cost = _setup()
    for epsilon in [0.5, 2.0, 5.0]:
        plan = sinkhorn(a, b, cost, epsilon=epsilon, n_iter=2000, tol=1e-14)
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


# --------------------- Limits of epsilon ----------------------------------- #
def test_sinkhorn_approaches_lp_optimum_at_small_epsilon():
    a, b, cost = _setup(n=5, m=5, seed=1)
    lp_cost = float(ot.emd2(a, b, cost))
    eps_small = 0.02
    sk_cost = sinkhorn_distance(a, b, cost, epsilon=eps_small, n_iter=2000)
    # Entropic cost relaxes the LP -> still close at small eps, but always >= LP cost
    # minus a bias proportional to eps * log(n*m).
    assert sk_cost == pytest.approx(lp_cost, rel=0.05, abs=0.05)


def test_sinkhorn_approaches_independent_coupling_at_large_epsilon():
    a, b, cost = _setup(n=4, m=4, seed=2)
    # The first-order correction is O(1/epsilon * max(C)); for max C ~ 5 and
    # epsilon = 1e5 the residual is ~ 5e-5.
    plan = sinkhorn(a, b, cost, epsilon=1e5, n_iter=2000)
    independent = np.outer(a, b)
    np.testing.assert_allclose(plan, independent, atol=1e-3)


def test_sinkhorn_matches_pot_sinkhorn():
    # Allow ~1e-3 absolute -- POT may use a slightly different update order, and the
    # fixed point is reached up to a known floating-point floor.
    a, b, cost = _setup(n=6, m=6, seed=3)
    epsilon = 0.3
    ours = sinkhorn(a, b, cost, epsilon=epsilon, n_iter=5000, tol=1e-14)
    theirs = ot.sinkhorn(a, b, cost, epsilon, numItermax=5000, stopThr=1e-14)
    np.testing.assert_allclose(ours, theirs, atol=1e-3)


# --------------------- Amari's KL-projection identity --------------------- #
def test_sinkhorn_plan_minimizes_kl_to_gibbs_kernel():
    """Amari's bridge: Sinkhorn plan = KL projection of K onto T(a, b).

    Any feasible perturbation that preserves marginals must increase KL(P || K).
    """
    a, b, cost = _setup(n=4, m=4, seed=4)
    epsilon = 0.4
    plan = sinkhorn(a, b, cost, epsilon=epsilon, n_iter=2000, tol=1e-12)
    kernel = gibbs_kernel(cost, epsilon)
    kl_at_optimum = kl_to_kernel(plan, kernel)

    # A 2x2 cycle perturbation E preserves both marginals (rows and cols sum to 0).
    rng = np.random.default_rng(0)
    for _ in range(8):
        i1, i2 = rng.choice(plan.shape[0], 2, replace=False)
        j1, j2 = rng.choice(plan.shape[1], 2, replace=False)
        E = np.zeros_like(plan)
        E[i1, j1] = 1.0
        E[i2, j2] = 1.0
        E[i1, j2] = -1.0
        E[i2, j1] = -1.0
        max_delta = min(plan[i1, j2], plan[i2, j1]) * 0.5
        if max_delta <= 1e-12:
            continue
        for delta in [-max_delta * 0.5, max_delta * 0.5]:
            perturbed = plan + delta * E
            if np.all(perturbed > 0):
                assert kl_to_kernel(perturbed, kernel) > kl_at_optimum - 1e-9


def test_kl_to_kernel_zero_iff_plan_equals_kernel():
    K = np.array([[0.2, 0.3], [0.4, 0.1]])
    assert kl_to_kernel(K, K) == pytest.approx(0.0, abs=1e-12)
    assert kl_to_kernel(np.array([[0.1, 0.2], [0.3, 0.4]]), K) > 0.0


# --------------------- Convergence diagnostic ------------------------------ #
def test_sinkhorn_iterations_log_decreases_marginal_errors_geometrically():
    # Use a sharp-but-safe epsilon so the kernel does not underflow.
    # Note: by construction the column update makes col-margins exact at every
    # iteration, so only the row-margin error has a non-trivial trajectory.
    a, b, cost = _setup(n=8, m=8, seed=5)
    err_row, _ = sinkhorn_iterations_log(a, b, cost, epsilon=0.5, n_iter=500)
    assert err_row[10] < err_row[0]
    assert err_row[-1] < err_row[10] / 100.0
    assert err_row[-1] < 1e-9
