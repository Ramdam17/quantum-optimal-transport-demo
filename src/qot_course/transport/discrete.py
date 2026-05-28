"""Discrete optimal transport: cost matrices, the Kantorovich LP, polytope checks.

S8 formalises the Monge--Kantorovich problem in the discrete setting. Given a source
histogram :math:`a \\in \\Delta^n` and target :math:`b \\in \\Delta^m` and a cost matrix
:math:`C \\in \\mathbb{R}_+^{n \\times m}`, the **Kantorovich linear program** is

.. math::
    \\min_{P \\in \\mathbb{R}_+^{n \\times m}} \\sum_{i, j} P_{ij}\\, C_{ij}
    \\quad \\text{s.t.} \\quad P\\,\\mathbf{1} = a, \\;\\; P^\\top \\mathbf{1} = b.

The feasible set is the **transportation polytope** :math:`T(a, b)`. When :math:`a = b =
\\mathbf{1}/n` (uniform, equal-size), it becomes the **Birkhoff polytope** of doubly
stochastic matrices --- whose extreme points are the :math:`n!` permutation matrices
(Birkhoff--von Neumann theorem) --- and the LP becomes the **assignment problem**.

References: G. Peyre and M. Cuturi, *Computational Optimal Transport* (NoW, 2019),
chs.~2--3; C. Villani, *Topics in Optimal Transportation* (AMS, 2003), ch.~1;
L. V. Kantorovich, "On the translocation of masses", Dokl. Akad. Nauk SSSR 37, 199 (1942).
"""

from __future__ import annotations

import numpy as np
import ot


# ----------------------------------------------------------------------------- #
# Cost matrices on 1-D ground spaces (the canonical cases for the course)
# ----------------------------------------------------------------------------- #
def squared_euclidean_cost(
    x_source: np.ndarray, x_target: np.ndarray
) -> np.ndarray:
    """Pairwise squared-distance cost ``C[i, j] = (x_source[i] - x_target[j])**2``.

    This is the ground cost of the 2-Wasserstein distance.
    """
    xs = np.asarray(x_source, dtype=float).reshape(-1, 1)
    xt = np.asarray(x_target, dtype=float).reshape(1, -1)
    return (xs - xt) ** 2


def cityblock_cost(x_source: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    """Pairwise absolute-difference cost ``C[i, j] = |x_source[i] - x_target[j]|``.

    Ground cost of the 1-Wasserstein (earth-mover's) distance.
    """
    xs = np.asarray(x_source, dtype=float).reshape(-1, 1)
    xt = np.asarray(x_target, dtype=float).reshape(1, -1)
    return np.abs(xs - xt)


# ----------------------------------------------------------------------------- #
# Kantorovich LP solver and total-cost accessor
# ----------------------------------------------------------------------------- #
def discrete_ot_plan(
    source: np.ndarray, target: np.ndarray, cost: np.ndarray
) -> np.ndarray:
    """Solve the discrete Kantorovich LP and return the optimal coupling ``P``.

    Wraps POT's network-simplex solver (:func:`ot.emd`). The returned plan has row
    sums equal to ``source`` and column sums equal to ``target`` (up to floating
    point); use :func:`is_transportation_polytope_member` to verify.
    """
    return ot.emd(
        np.asarray(source, dtype=float),
        np.asarray(target, dtype=float),
        np.asarray(cost, dtype=float),
    )


def transport_cost(plan: np.ndarray, cost: np.ndarray) -> float:
    """Total transport cost ``<C, P> = sum_{i, j} P[i, j] * C[i, j]``."""
    plan = np.asarray(plan, dtype=float)
    cost = np.asarray(cost, dtype=float)
    return float(np.sum(plan * cost))


# ----------------------------------------------------------------------------- #
# Polytope membership: marginal constraints and the Birkhoff case
# ----------------------------------------------------------------------------- #
def is_transportation_polytope_member(
    plan: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    atol: float = 1e-9,
) -> bool:
    """Check membership in the transportation polytope ``T(source, target)``.

    Returns ``True`` iff ``plan`` has the right shape, non-negative entries, and
    marginals matching ``source`` (rows) and ``target`` (columns).
    """
    plan = np.asarray(plan, dtype=float)
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if plan.shape != (len(source), len(target)):
        return False
    rows_ok = bool(np.allclose(plan.sum(axis=1), source, atol=atol))
    cols_ok = bool(np.allclose(plan.sum(axis=0), target, atol=atol))
    nonneg = bool(np.all(plan >= -atol))
    return rows_ok and cols_ok and nonneg


def is_doubly_stochastic(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    """Check membership in the Birkhoff polytope of doubly stochastic matrices.

    Requires square shape, row sums equal to 1, column sums equal to 1, and
    non-negative entries.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    rows_one = bool(np.allclose(matrix.sum(axis=1), 1.0, atol=atol))
    cols_one = bool(np.allclose(matrix.sum(axis=0), 1.0, atol=atol))
    nonneg = bool(np.all(matrix >= -atol))
    return rows_one and cols_one and nonneg
