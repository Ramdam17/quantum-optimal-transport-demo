"""Wasserstein distances: LP-based general case and the 1-D closed form.

S9 promotes the Kantorovich LP cost to a genuine *metric* on probability distributions
--- the Wasserstein-:math:`p` distance ---

.. math::
    W_p(\\mu, \\nu) = \\left( \\inf_{\\pi \\in \\Pi(\\mu, \\nu)}
                     \\int c(x, y)^p\\, \\mathrm{d}\\pi(x, y) \\right)^{1/p},

with :math:`c(x, y) = |x - y|` for points in :math:`\\mathbb{R}^d`. On a 1-D ground space
the LP collapses to *sort-and-match by quantile* (Brenier; Villani, 2003, Thm.~2.18):

.. math::
    W_p^p(\\mu, \\nu) = \\int_0^1 \\bigl|F_\\mu^{-1}(u) - F_\\nu^{-1}(u)\\bigr|^p\\,
                       \\mathrm{d}u,

which for discrete atomic distributions is a finite sum over the *common refinement* of
the two CDFs --- computable in :math:`\\mathcal{O}((n + m)\\log(n + m))` instead of
:math:`\\mathcal{O}(n^3)` for the LP. The closed form vanishes in higher dimensions
because the quantile function does.

References: C. Villani, *Topics in Optimal Transportation* (AMS, 2003), ch.~2; G. Peyre
and M. Cuturi, *Computational Optimal Transport* (NoW, 2019), ch.~2; R. McCann, "A
convexity principle for interacting gases", Adv. Math. 128, 153 (1997).
"""

from __future__ import annotations

import numpy as np

from qot_course.transport.discrete import discrete_ot_plan, transport_cost


def wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    p: float = 2.0,
) -> float:
    """Wasserstein-:math:`p` distance computed via the Kantorovich LP.

    The cost matrix passed here must already have entries :math:`c(x_i, y_j)^p`
    (POT's convention). The function returns :math:`(\\langle C, P^\\star \\rangle)^{1/p}`
    where :math:`P^\\star` is the LP optimum.
    """
    plan = discrete_ot_plan(source, target, cost)
    return float(transport_cost(plan, cost) ** (1.0 / p))


def wasserstein_1d(
    positions_a: np.ndarray,
    weights_a: np.ndarray,
    positions_b: np.ndarray,
    weights_b: np.ndarray,
    p: float = 2.0,
) -> float:
    """Exact 1-D Wasserstein-:math:`p` distance via quantile sort-and-match.

    Computes
    :math:`\\bigl(\\int_0^1 |F_a^{-1}(u) - F_b^{-1}(u)|^p\\,\\mathrm{d}u\\bigr)^{1/p}`
    by summing over the common refinement of the two CDFs. Both atomic distributions
    must have the same total mass.

    Parameters
    ----------
    positions_a, positions_b : array-like
        Real positions of the atoms in each distribution.
    weights_a, weights_b : array-like
        Non-negative weights summing to the same total mass on both sides.
    p : float
        Order of the Wasserstein distance (``p >= 1``).
    """
    pa = np.asarray(positions_a, dtype=float).ravel()
    wa = np.asarray(weights_a, dtype=float).ravel()
    pb = np.asarray(positions_b, dtype=float).ravel()
    wb = np.asarray(weights_b, dtype=float).ravel()
    if pa.shape != wa.shape or pb.shape != wb.shape:
        raise ValueError("positions and weights must have matching shapes.")
    if abs(wa.sum() - wb.sum()) > 1e-9:
        raise ValueError("Source and target total masses must agree.")
    # Normalise defensively to total mass 1.
    wa = wa / wa.sum()
    wb = wb / wb.sum()

    # Sort each side by position.
    order_a = np.argsort(pa)
    order_b = np.argsort(pb)
    pa, wa = pa[order_a], wa[order_a]
    pb, wb = pb[order_b], wb[order_b]

    cdf_a = np.cumsum(wa)  # right-continuous CDF values at the n atoms
    cdf_b = np.cumsum(wb)

    # Common refinement of the two CDFs, with 0 as the left anchor.
    breakpoints = np.unique(
        np.concatenate([[0.0], cdf_a, cdf_b, [1.0]])
    )
    breakpoints = breakpoints[(breakpoints >= 0.0) & (breakpoints <= 1.0)]

    total = 0.0
    for k in range(len(breakpoints) - 1):
        u_lo, u_hi = breakpoints[k], breakpoints[k + 1]
        if u_hi <= u_lo:
            continue
        # Use a point strictly inside (u_lo, u_hi) to locate the constant quantile values.
        u_mid = 0.5 * (u_lo + u_hi)
        i = int(np.searchsorted(cdf_a, u_mid, side="left"))
        j = int(np.searchsorted(cdf_b, u_mid, side="left"))
        i = min(i, len(pa) - 1)
        j = min(j, len(pb) - 1)
        total += (u_hi - u_lo) * abs(pa[i] - pb[j]) ** p

    return float(total ** (1.0 / p))


def mccann_geodesic_atoms_1d(
    positions_a: np.ndarray,
    weights_a: np.ndarray,
    positions_b: np.ndarray,
    weights_b: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic McCann (displacement) geodesic at time ``t`` between two 1-D distributions.

    Returns ``(positions_t, weights_t)`` of an atomic representation of
    :math:`\\mu_t = ((1-t)\\,\\mathrm{Id} + t\\,T)_{\\#}\\mu`, where :math:`T = F_\\nu^{-1}
    \\circ F_\\mu` is the optimal transport map. The output may have *more* atoms than
    either input --- one per breakpoint of the common CDF refinement.
    """
    pa = np.asarray(positions_a, dtype=float).ravel()
    wa = np.asarray(weights_a, dtype=float).ravel()
    pb = np.asarray(positions_b, dtype=float).ravel()
    wb = np.asarray(weights_b, dtype=float).ravel()
    if abs(wa.sum() - wb.sum()) > 1e-9:
        raise ValueError("Source and target total masses must agree.")
    wa = wa / wa.sum()
    wb = wb / wb.sum()
    pa, wa = pa[np.argsort(pa)], wa[np.argsort(pa)]
    pb, wb = pb[np.argsort(pb)], wb[np.argsort(pb)]
    cdf_a = np.cumsum(wa)
    cdf_b = np.cumsum(wb)
    breakpoints = np.unique(np.concatenate([[0.0], cdf_a, cdf_b, [1.0]]))

    positions_t: list[float] = []
    weights_t: list[float] = []
    for k in range(len(breakpoints) - 1):
        u_lo, u_hi = breakpoints[k], breakpoints[k + 1]
        if u_hi <= u_lo:
            continue
        u_mid = 0.5 * (u_lo + u_hi)
        i = min(int(np.searchsorted(cdf_a, u_mid, side="left")), len(pa) - 1)
        j = min(int(np.searchsorted(cdf_b, u_mid, side="left")), len(pb) - 1)
        x_t = (1.0 - t) * pa[i] + t * pb[j]
        positions_t.append(x_t)
        weights_t.append(u_hi - u_lo)
    return np.asarray(positions_t), np.asarray(weights_t)
