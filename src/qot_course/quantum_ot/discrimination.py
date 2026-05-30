r"""The discriminating experiment --- two ensembles PLV cannot tell apart, QOT can (Plan F, 04/15).

This module is the **scientific crux** of the course: it builds two synthetic dyads with
**identical phase-locking value by construction** but **different higher-order phase
structure**, and shows that PLV (and the naive phase-qubit embedding) cannot tell them
apart while a richer multi-frequency embedding's quantum mutual information can.

The mechanism is *higher-order phase coupling*. We draw the phase difference
:math:`\delta = \theta_A - \theta_B` from the truncated-Fourier circular density

.. math::

    p(\delta) \;\propto\; 1 + 2\,a_1 \cos\delta + 2\,a_2 \cos 2\delta ,
    \qquad \delta \in [0, 2\pi),

whose first two circular (trigonometric) moments are *exactly*
:math:`\langle e^{i\delta}\rangle = a_1` and :math:`\langle e^{2i\delta}\rangle = a_2`
(Mardia & Jupp, ch. 3: the order-:math:`k` Fourier coefficient of a wrapped density is its
:math:`k`-th circular moment). Building two ensembles with the **same** :math:`a_1` fixes
PLV identically; giving them **different** :math:`a_2` changes only the second moment.

A shared, uniform :math:`\theta_A` with :math:`\theta_B = \theta_A - \delta` keeps both
marginals (very nearly) maximally mixed, so the two ensembles differ **only** in the joint
higher-order structure. The naive phase qubit
:math:`(|0\rangle + e^{i\theta}|1\rangle)/\sqrt2` carries a coherence proportional to the
*first* moment and is therefore blind to :math:`a_2`; the multi-frequency embedding
:func:`~qot_course.quantum_ot.embeddings.multifreq_state` additionally carries :math:`a_2`
in the :math:`|2\rangle_A|0\rangle_B \leftrightarrow |0\rangle_A|2\rangle_B` coherence, and
its quantum mutual information *sees* the difference.

References
----------
K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
DOI:10.1002/9780470316979.
J.-P. Lachaux, E. Rodriguez, J. Martinerie, F. J. Varela, "Measuring phase synchrony in
brain signals", *Hum. Brain Mapp.* **8**, 194 (1999),
DOI:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _two_harmonic_density(
    delta: NDArray[np.float64], a1: float, a2: float
) -> NDArray[np.float64]:
    r"""Evaluate the unnormalised truncated-Fourier density ``1 + 2 a1 cos δ + 2 a2 cos 2δ``.

    Internal helper for :func:`sample_phase_difference`. The density is unnormalised (its
    integral over :math:`[0, 2\pi)` is :math:`2\pi`); rejection sampling only needs it up to
    a constant. It is **nonnegative iff** :math:`\min_\delta(1 + 2a_1\cos\delta + 2a_2\cos
    2\delta) \ge 0`; this holds for the parameters used in the discriminating experiment
    (``a1=0.4`` with ``a2 in {0.0, 0.3}`` give minima ``0.20`` and ``0.267`` respectively).

    Parameters
    ----------
    delta : ndarray of float, shape (N,)
        Phase differences in radians at which to evaluate the density.
    a1, a2 : float
        First and second Fourier coefficients (target first/second circular moments).

    Returns
    -------
    ndarray of float, shape (N,)
        Unnormalised density values at each ``delta``.
    """
    return 1.0 + 2.0 * a1 * np.cos(delta) + 2.0 * a2 * np.cos(2.0 * delta)


def sample_phase_difference(
    n: int, a1: float, a2: float, *, seed: int
) -> NDArray[np.float64]:
    r"""Sample phase differences from ``p(δ) ∝ 1 + 2 a1 cos δ + 2 a2 cos 2δ`` by rejection.

    Draws ``n`` i.i.d. samples of the phase difference :math:`\delta \in [0, 2\pi)` from the
    truncated-Fourier circular density. By construction the *normalised* density has circular
    moments :math:`\langle e^{i\delta}\rangle = a_1` (so :math:`\langle\cos\delta\rangle =
    a_1`, :math:`\langle\sin\delta\rangle = 0`) and :math:`\langle e^{2i\delta}\rangle = a_2`.
    The coefficient ``a1`` sets first-order phase concentration (what PLV measures); ``a2``
    sets the **second**-order structure (anti-phase / bimodal symmetry) that PLV is blind to.

    Sampling is by simple rejection against a uniform envelope: the density is bounded above
    by ``peak = 1 + 2|a1| + 2|a2|``, so each candidate :math:`\delta \sim U[0, 2\pi)` is kept
    with probability ``p(δ) / peak``. The acceptance rate is ``1 / peak`` (e.g. ``≈ 0.36`` for
    ``a1=0.4, a2=0.3``), so the loop refills until ``n`` samples are collected.

    Parameters
    ----------
    n : int
        Number of phase-difference samples to draw.
    a1 : float
        Target first circular moment :math:`\langle\cos\delta\rangle`. For PLV-matched
        ensembles this is held fixed across conditions.
    a2 : float
        Target second circular moment :math:`\langle\cos 2\delta\rangle`. Varying this (with
        ``a1`` fixed) is the higher-order structure PLV cannot see.
    seed : int, keyword-only
        Seed for the NumPy ``default_rng`` generator (reproducibility).

    Returns
    -------
    delta : ndarray of float, shape (n,)
        Phase-difference samples in radians, each in :math:`[0, 2\pi)`.

    Notes
    -----
    When to use
        Use to generate phase differences with *prescribed* first and second circular
        moments --- the building block of :func:`matched_plv_ensembles` and the 04/15
        discriminating experiment. When you only need a unimodal phase distribution with a
        single concentration parameter, the von Mises sampler
        (``numpy.random.Generator.vonmises``) is simpler; this two-harmonic density is the
        right tool precisely when you must **control the second moment independently** of the
        first.

    Nonnegativity
        The density must satisfy :math:`p(\delta) \ge 0` everywhere to be a valid probability
        density. This holds for the experiment's parameters (``a1=0.4``, ``a2 in {0.0,
        0.3}``); for arbitrary ``(a1, a2)`` it is **not** guaranteed (e.g. large ``|a1|`` can
        drive the minimum negative). Verify
        ``_two_harmonic_density(np.linspace(0, 2*np.pi, 10001), a1, a2).min() >= 0`` before
        trusting the moments for new parameter choices --- a negative envelope would make the
        rejection step sample from ``max(p, 0)`` instead, silently distorting the moments.

    Examples
    --------
    >>> import numpy as np
    >>> d = sample_phase_difference(80000, a1=0.4, a2=0.3, seed=0)
    >>> d.shape
    (80000,)
    >>> bool(abs(float(np.mean(np.cos(d))) - 0.4) < 0.02)      # first moment ~ a1
    True
    >>> bool(abs(float(np.mean(np.cos(2 * d))) - 0.3) < 0.02)  # second moment ~ a2
    True

    References
    ----------
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    """
    rng = np.random.default_rng(seed)
    peak = 1.0 + 2.0 * abs(a1) + 2.0 * abs(a2)  # the density is bounded above by this
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(0.0, 2.0 * np.pi, size=n)
        accept = rng.uniform(0.0, peak, size=n) < _two_harmonic_density(cand, a1, a2)
        keep = cand[accept]
        take = min(keep.size, n - filled)
        out[filled : filled + take] = keep[:take]
        filled += take
    return out


def matched_plv_ensembles(
    n: int, a1: float, a2_low: float, a2_high: float, *, seed: int
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    r"""Build two dyads with identical PLV but different second circular moment.

    Generates two synthetic two-channel ("dyad") phase records that share the **same** first
    circular moment :math:`a_1` --- hence (in the large-``n`` limit) the **same PLV** --- but
    have **different** second moments :math:`a_2`. This is the codified thesis of the course:
    a difference that PLV (and the naive phase-qubit embedding) cannot detect, but a
    multi-frequency quantum embedding can.

    Both dyads share a single uniform reference phase
    :math:`\theta_A \sim U[0, 2\pi)` and set :math:`\theta_B = \theta_A - \delta`, where
    :math:`\delta` is drawn from the two-harmonic density (:func:`sample_phase_difference`)
    with the *same* ``a1`` but the respective ``a2_low`` / ``a2_high``. Because :math:`\theta_A`
    is uniform and :math:`\delta` is independent of it, both single-channel marginals are very
    nearly uniform (maximally mixed): the two dyads therefore differ **only** in their joint
    higher-order structure, isolating the effect of :math:`a_2`.

    Parameters
    ----------
    n : int
        Number of samples per channel in each dyad.
    a1 : float
        Shared first circular moment of :math:`\delta` --- fixes PLV equal across both dyads.
    a2_low : float
        Second circular moment of the **low** dyad (e.g. ``0.0`` --- no second-order
        structure).
    a2_high : float
        Second circular moment of the **high** dyad (e.g. ``0.3`` --- nonzero second-order
        structure). With ``a1=0.4``, both ``a2_low=0.0`` and ``a2_high=0.3`` keep the density
        nonnegative (minima ``0.20`` and ``0.267``).
    seed : int, keyword-only
        Master seed. The reference phase :math:`\theta_A` uses ``seed``; the low/high phase
        differences use ``seed + 1`` / ``seed + 2`` so the two dyads share :math:`\theta_A`
        but draw independent :math:`\delta`.

    Returns
    -------
    low : tuple of (ndarray, ndarray), each shape (n,)
        ``(theta_A, theta_B_low)`` for the low-:math:`a_2` dyad, phases in radians.
    high : tuple of (ndarray, ndarray), each shape (n,)
        ``(theta_A, theta_B_high)`` for the high-:math:`a_2` dyad. The same ``theta_A`` array
        is shared with ``low``.

    Notes
    -----
    When to use
        Use to produce the two ensembles compared in notebook 04/15 (and the thesis test):
        feed each ``(theta_A, theta_B)`` pair to :func:`~qot_course.quantum_ot.capstone.plv`
        (matched), to the naive
        :func:`~qot_course.quantum_ot.capstone.joint_density_matrix` +
        :func:`~qot_course.quantum_ot.capstone.coupling_qmi` (also matched --- the naive
        coherence is the first moment only), and to
        :func:`~qot_course.quantum_ot.embeddings.multifreq_state` +
        :func:`~qot_course.quantum_ot.embeddings.joint_density_from_states` +
        ``coupling_qmi(..., dims=(3, 3))`` (separated --- the qutrit carries :math:`a_2`).
        When you instead want a *graded* coupling sweep (PLV and QMI both rising with one
        knob), use :func:`~qot_course.quantum_ot.capstone.sweep_coupling_measures`; this
        function is for the orthogonal *discrimination* question.

    Use a large ``n`` (the experiment uses ``80000``): PLV and the circular moments are
    Monte-Carlo estimates whose error shrinks like :math:`1/\sqrt{n}`, and the thesis requires
    the *matched* quantities (PLV, naive QMI) to agree to within ``0.02`` while the *separated*
    quantity (rich QMI) differs by much more.

    Examples
    --------
    >>> import numpy as np
    >>> from qot_course.quantum_ot.capstone import plv
    >>> (ta, tb_low), (ta2, tb_high) = matched_plv_ensembles(
    ...     n=80000, a1=0.4, a2_low=0.0, a2_high=0.3, seed=0
    ... )
    >>> bool(np.array_equal(ta, ta2))             # the reference phase is shared
    True
    >>> bool(abs(plv(ta, tb_low) - plv(ta2, tb_high)) < 0.02)   # PLV matched by construction
    True

    References
    ----------
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    J.-P. Lachaux et al., "Measuring phase synchrony in brain signals",
    *Hum. Brain Mapp.* **8**, 194 (1999),
    DOI:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C.
    """
    rng = np.random.default_rng(seed)
    theta_a = rng.uniform(0.0, 2.0 * np.pi, size=n)
    d_low = sample_phase_difference(n, a1, a2_low, seed=seed + 1)
    d_high = sample_phase_difference(n, a1, a2_high, seed=seed + 2)
    low = (theta_a, np.mod(theta_a - d_low, 2.0 * np.pi))
    high = (theta_a, np.mod(theta_a - d_high, 2.0 * np.pi))
    return low, high
