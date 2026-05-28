"""Information geometry: Fisher metric, statistical manifolds, geodesics on the simplex.

The simplex of categorical distributions carries (at least) two natural geometries:

- the **information geometry** of Fisher--Rao / KL --- the natural geometry of
  *statistical* closeness (how distinguishable two distributions are from samples);
- the **transport geometry** of Wasserstein --- the natural geometry of *spatial*
  closeness (how far mass has to move on the ground space).

The two answer different questions; this module gives the elementary tools to compute
and compare them. References: S.-i. Amari, *Information Geometry and Its Applications*
(Springer, 2016); F. Nielsen, "An elementary introduction to information geometry",
arXiv:1808.08271 (2020); G. Peyre and M. Cuturi, *Computational Optimal Transport*
(NoW, 2019), ch. 7.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- #
# Fisher information metric
# ----------------------------------------------------------------------------- #
def bernoulli_fisher(theta: float) -> float:
    """Fisher information of the Bernoulli family at parameter ``theta``.

    For ``x in {0, 1}`` with ``p(x=1) = theta`` and ``p(x=0) = 1 - theta``, the Fisher
    information is

    .. math::
        I(\\theta) = \\mathbb{E}\\!\\left[\\left(\\frac{\\partial \\log p}{\\partial \\theta}\\right)^2\\right]
        = \\frac{1}{\\theta(1-\\theta)}.

    The metric blows up at the boundary --- the corners of the simplex are infinitely
    far away in Fisher--Rao distance.

    Parameters
    ----------
    theta : float
        Bernoulli parameter, strictly inside ``(0, 1)``.
    """
    if not 0.0 < theta < 1.0:
        raise ValueError(f"theta must be in (0, 1); got {theta}")
    return 1.0 / (theta * (1.0 - theta))


def multinomial_fisher(p: np.ndarray) -> np.ndarray:
    """Fisher information matrix of the categorical / multinomial family at ``p``.

    In the *raw* probability parametrisation (no constraint reduction), the FIM of
    the categorical family is the diagonal matrix ``diag(1 / p_i)`` (Amari, 2016,
    sec. 2.2). The constraint ``sum p_i = 1`` only enters when we restrict to
    tangent vectors that respect it.

    Parameters
    ----------
    p : np.ndarray
        Strictly positive probability vector (each ``p_i > 0``).
    """
    p = np.asarray(p, dtype=float).ravel()
    if np.any(p <= 0):
        raise ValueError("Fisher metric requires strictly positive probabilities.")
    return np.diag(1.0 / p)


# ----------------------------------------------------------------------------- #
# Fisher--Rao distance and geodesic on the simplex
# ----------------------------------------------------------------------------- #
def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher--Rao geodesic distance between two categorical distributions.

    The change of variable ``phi_i = sqrt(p_i)`` is an isometry from the simplex
    (with the Fisher--Rao metric) onto the unit sphere; the geodesic distance is the
    angle between the sqrt-vectors (Rao, 1945; Bhattacharyya, 1943):

    .. math::
        d_{\\text{FR}}(p, q) = 2\\,\\arccos\\!\\left(\\sum_i \\sqrt{p_i\\, q_i}\\right)
        \\in [0, \\pi].

    Two distributions are at the maximum distance ``pi`` iff their supports are disjoint.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    inner = float(np.sum(np.sqrt(np.clip(p, 0.0, None) * np.clip(q, 0.0, None))))
    # Clip for numerical safety against arccos domain errors at the boundary.
    inner = max(-1.0, min(1.0, inner))
    return 2.0 * float(np.arccos(inner))


def fisher_rao_geodesic(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Point along the Fisher--Rao geodesic at time ``t`` (``0`` at ``p``, ``1`` at ``q``).

    Geodesics on the sphere of sqrt-distributions are great-circle arcs (slerp):

    .. math::
        \\phi_t = \\frac{\\sin((1 - t)\\Theta)}{\\sin\\Theta}\\sqrt{p}
                + \\frac{\\sin(t\\Theta)}{\\sin\\Theta}\\sqrt{q},
        \\qquad \\Theta = \\arccos\\!\\big(\\textstyle\\sum_i \\sqrt{p_i q_i}\\big).

    Squaring elementwise and renormalising returns the geodesic distribution ``p_t``.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    sp, sq = np.sqrt(np.clip(p, 0.0, None)), np.sqrt(np.clip(q, 0.0, None))
    cos_theta = float(np.sum(sp * sq))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        # Nearly identical: avoid 0/0 in slerp; fall back to linear sqrt-interp.
        phi_t = (1.0 - t) * sp + t * sq
    else:
        phi_t = (
            np.sin((1.0 - t) * theta) * sp + np.sin(t * theta) * sq
        ) / np.sin(theta)
    p_t = phi_t**2
    return p_t / p_t.sum()


# ----------------------------------------------------------------------------- #
# Interpolations: the two geometries on a 1-D ground space
# ----------------------------------------------------------------------------- #
def mixture_interpolation(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Linear (m-geodesic) interpolation ``(1 - t) p + t q``.

    This is the *mixture* geodesic of Amari's mixture (m-) connection on the simplex.
    It is "vertical": it blends masses bin by bin and ignores the ground space. Between
    two well-separated bumps on a line, the midpoint is *bimodal*.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    return (1.0 - t) * p + t * q


def wasserstein_interpolation_1d(
    p: np.ndarray,
    q: np.ndarray,
    support: np.ndarray,
    t: float,
    n_particles: int = 1000,
) -> np.ndarray:
    """McCann displacement interpolation between two 1-D discrete distributions.

    In one dimension the optimal map is the quantile composition ``Q_q . F_p`` (Brenier,
    1991): each ``p``-mass element at quantile ``u`` moves to the ``q``-position with
    the same quantile ``u``. We approximate the McCann interpolation by sampling
    ``n_particles`` equi-mass particles via inverse CDFs, sliding each from its
    ``p``-position to its ``q``-position by a fraction ``t``, then rebinning on the
    original support.

    Parameters
    ----------
    p, q : np.ndarray
        Probability mass functions on the shared 1-D ``support``.
    support : np.ndarray
        Strictly increasing grid of positions (length ``len(p) == len(q)``).
    t : float
        Time along the geodesic, ``0`` at ``p`` and ``1`` at ``q``.
    n_particles : int
        Number of equi-mass particles used in the quantile-coupling rebinning.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    support = np.asarray(support, dtype=float).ravel()
    if p.shape != q.shape or p.shape != support.shape:
        raise ValueError("p, q, and support must have the same shape.")

    cdf_p = np.cumsum(p)
    cdf_p = cdf_p / cdf_p[-1]
    cdf_q = np.cumsum(q)
    cdf_q = cdf_q / cdf_q[-1]

    u = (np.arange(n_particles) + 0.5) / n_particles
    xp = np.interp(u, cdf_p, support)
    xq = np.interp(u, cdf_q, support)
    xt = (1.0 - t) * xp + t * xq

    # Bin edges around the support points (midpoints, with half-step margins on the ends).
    edges = np.concatenate(
        [
            [support[0] - 0.5 * (support[1] - support[0])],
            0.5 * (support[:-1] + support[1:]),
            [support[-1] + 0.5 * (support[-1] - support[-2])],
        ]
    )
    counts, _ = np.histogram(xt, bins=edges)
    counts = counts.astype(float)
    return counts / counts.sum()


# ----------------------------------------------------------------------------- #
# Simplex plotting coordinates
# ----------------------------------------------------------------------------- #
def simplex_to_cartesian(p: np.ndarray) -> np.ndarray:
    """Map a 3-vector on the 2-simplex to 2-D barycentric Cartesian coordinates.

    Vertices placed at ``delta_1 = (0, 0)``, ``delta_2 = (1, 0)``, ``delta_3 = (0.5,
    sqrt(3)/2)`` so the simplex appears as an equilateral triangle.
    """
    p = np.asarray(p, dtype=float).ravel()
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    return p[0] * v0 + p[1] * v1 + p[2] * v2
