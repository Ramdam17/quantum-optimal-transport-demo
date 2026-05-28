"""Entropic optimal transport: the Sinkhorn algorithm and Amari's KL bridge.

S10 regularises the Kantorovich LP by an entropy bonus:

.. math::
    P^\\star_\\varepsilon = \\arg\\min_{P \\in T(a, b)}\\
       \\langle C, P\\rangle - \\varepsilon\\, H(P),
    \\qquad H(P) = -\\sum_{i, j} P_{ij}\\log P_{ij}.

A one-line algebra reveals **Amari's bridge**: with the Gibbs kernel
:math:`K_{ij} = \\exp(-C_{ij}/\\varepsilon)`,

.. math::
    \\varepsilon\\, \\mathrm{KL}(P \\,\\|\\, K)
    = \\sum_{ij} P_{ij}\\bigl(\\varepsilon\\log P_{ij} + C_{ij}\\bigr) + \\text{const}
    = \\langle C, P\\rangle - \\varepsilon H(P) + \\text{const},

so the entropic-OT plan is the **KL projection** of the Gibbs kernel onto the
transportation polytope :math:`T(a, b)`. This is exactly where Wasserstein (M3) meets
KL / Fisher--Rao (M2).

The Sinkhorn algorithm (Sinkhorn 1964; rediscovered by Cuturi 2013 for OT) is the
*iterative Bregman projection* onto the two marginal constraints, which under the KL
Bregman divergence is just alternating multiplicative rescaling of rows and columns.

References: M. Cuturi, "Sinkhorn distances: lightspeed computation of optimal transport",
NeurIPS (2013); G. Peyre and M. Cuturi, *Computational Optimal Transport* (NoW, 2019),
chs.~4--5; S.-i. Amari, *Information Geometry and Its Applications* (Springer, 2016),
sec.~7.5; R. Sinkhorn, "A relationship between arbitrary positive matrices and doubly
stochastic matrices", Ann. Math. Statist. 35, 876 (1964).
"""

from __future__ import annotations

import numpy as np


def gibbs_kernel(cost: np.ndarray, epsilon: float) -> np.ndarray:
    """Gibbs kernel ``K[i, j] = exp(-C[i, j] / epsilon)``.

    Centre point of the KL projection: the unconstrained minimiser of
    ``epsilon * KL(P || K) - <C, P>`` is ``K`` itself (without marginals).
    Adding the marginal constraints gives the entropic OT problem.
    """
    return np.exp(-np.asarray(cost, dtype=float) / epsilon)


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Sinkhorn algorithm for entropic optimal transport.

    Returns the optimal plan :math:`P^\\star_\\varepsilon` minimising
    :math:`\\langle C, P\\rangle - \\varepsilon H(P)` over the transportation polytope
    :math:`T(a, b)`. Equivalently, the KL projection of the Gibbs kernel
    :math:`K = \\exp(-C/\\varepsilon)` onto :math:`T(a, b)`.

    Parameters
    ----------
    a, b : array-like
        Source and target probability vectors.
    cost : array-like
        Cost matrix (entries non-negative).
    epsilon : float
        Entropic regularisation strength. Small :math:`\\varepsilon` -> sharp plan
        (close to the LP optimum). Large :math:`\\varepsilon` -> blurry plan (close
        to the independent coupling :math:`a\\otimes b`).
    n_iter, tol : int, float
        Maximum iterations and convergence tolerance on the max change in ``u``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    K = gibbs_kernel(cost, epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        v = b / np.maximum(K.T @ u, 1e-300)
        u_new = a / np.maximum(K @ v, 1e-300)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new
    v = b / np.maximum(K.T @ u, 1e-300)
    return (u[:, None] * K) * v[None, :]


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> float:
    """Entropic transport cost ``<C, P_eps>`` for the Sinkhorn plan.

    Note: this is *not* a proper distance (no triangle inequality at fixed
    :math:`\\varepsilon > 0`); the *debiased* Sinkhorn divergence
    :math:`S_\\varepsilon(\\mu, \\nu) = \\mathrm{OT}_\\varepsilon(\\mu, \\nu)
    - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\mu, \\mu)
    - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\nu, \\nu)` (Feydy et al., 2019) does.
    """
    plan = sinkhorn(a, b, cost, epsilon, n_iter, tol)
    return float(np.sum(plan * np.asarray(cost, dtype=float)))


def sinkhorn_iterations_log(
    a: np.ndarray, b: np.ndarray, cost: np.ndarray, epsilon: float, n_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Per-iteration row- and column-marginal max-errors of Sinkhorn (for visualisation).

    Returns ``(err_row, err_col)`` arrays of length ``n_iter`` showing how the marginal
    constraints close geometrically.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    K = gibbs_kernel(cost, epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)
    err_row = np.zeros(n_iter)
    err_col = np.zeros(n_iter)
    for k in range(n_iter):
        plan = (u[:, None] * K) * v[None, :]
        err_row[k] = float(np.max(np.abs(plan.sum(axis=1) - a)))
        err_col[k] = float(np.max(np.abs(plan.sum(axis=0) - b)))
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
    return err_row, err_col


def kl_to_kernel(plan: np.ndarray, kernel: np.ndarray) -> float:
    """Generalised KL divergence ``KL(P || K) = sum P log(P/K) - sum P + sum K``.

    For *unnormalised* positive matrices this is the right Bregman divergence; on the
    transportation polytope (where the linear-mass terms are constant) the minimisers
    coincide with those of the strict KL.
    """
    plan = np.asarray(plan, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    mask = plan > 0
    log_ratio = np.zeros_like(plan)
    log_ratio[mask] = np.log(plan[mask] / kernel[mask])
    return float(np.sum(plan * log_ratio) - np.sum(plan) + np.sum(kernel))
