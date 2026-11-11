"""Statistics of the free-second-harmonic bench, on one record and exactly.

Every comparison of the bench reads the same list of statistics from a pair of
phase records: the classical phase-locking values of the first and second
harmonics of the phase difference, the two components of the relative phase
of the first two circular moments, and quantum mutual information and Bures
distance of joint densities built from three kinds of phase embedding. This
module computes that list on one simulated record
(:func:`record_statistics`) and, from the exact stationary circular moments of
:mod:`qot_course.quantum_ot.biharmonic`, the long-run value of the same list
(:func:`long_run_statistics`). The two functions return the same keys in the
same order, and each key is the same quantity computed two ways: the same
embedding, the same subsystem dimensions, nats for information, and the sign
convention :math:`\\Delta = \\theta_1 - \\theta_2`,
:math:`m_k = E[e^{ik\\Delta}]`.

The module also holds the protocol that compares two coupling settings
(:func:`compare_conditions`). It draws independent record sets, reduces each to
its statistics, and summarises the difference between the settings three ways:
a signed score against the scatter of a null made of two sets of the same
setting (:func:`signed_score`), the area under the curve of the raw values
(:func:`auc`), and a two-sided detection rate (:func:`detection_rate`).

References
----------
G. Gatto, S. R. Jammalamadaka, "The generalized von Mises distribution",
*Stat. Methodol.* **4**, 341 (2007). DOI: 10.1016/j.stamet.2006.11.003.

J. A. Hanley, B. J. McNeil, "The meaning and use of the area under a receiver
operating characteristic (ROC) curve", *Radiology* **143**, 29 (1982).
DOI: 10.1148/radiology.143.1.7063747.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from qot_course.quantum_ot.biharmonic import (
    long_run_joint_density,
    simulate_biharmonic_dyad,
    stationary_moments,
)
from qot_course.quantum_ot.capstone import (
    coupling_bures,
    coupling_qmi,
    joint_density_matrix,
    plv,
)
from qot_course.quantum_ot.embeddings import (
    joint_density_from_states,
    multifreq_state,
    weighted_multifreq_state,
)
from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)

# Keys common to every statistics dictionary, in the order they are returned.
STATISTIC_KEYS_BASE = (
    "plv",
    "plv2",
    "relphase_sin",
    "relphase_cos",
    "qmi_naive",
    "qmi_rich",
    "bures_rich",
)
# Harmonics of the naive embedding: the phase qubit of capstone.joint_density_matrix,
# (|0> + e^{i theta}|1>)/sqrt(2), is the equal-weight embedding with harmonic 1
# (structural).
_NAIVE_HARMONICS = (1,)
# Harmonics of the rich and weighted embeddings: the qutrit carrying the first and
# second circular moments (structural, fixed by the bench design).
_RICH_HARMONICS = (1, 2)
# Subsystem dimensions passed to coupling_qmi and coupling_bures, one plus the
# number of harmonics of each embedding (derived).
_NAIVE_DIMS = (1 + len(_NAIVE_HARMONICS), 1 + len(_NAIVE_HARMONICS))
_RICH_DIMS = (1 + len(_RICH_HARMONICS), 1 + len(_RICH_HARMONICS))
# Highest circular moment the exact statistics read: long_run_joint_density needs
# entries up to the largest gap |h_i - h_k| between channel harmonics, which is
# max(_RICH_HARMONICS) because the DC channel has harmonic 0 (derived).
_MAX_EXACT_HARMONIC = max(_RICH_HARMONICS)
# A mean, a standard deviation with ddof=1 or a null band needs at least two
# finite values (structural: the sample standard deviation is undefined below two).
_MIN_FINITE_VALUES = 2
# Number of independent record sets drawn by compare_conditions, in the order
# A, B, A', A'', B', B'' (structural, fixed by the bench design).
_N_RECORD_SETS = 6
# Keys of the model mapping of compare_conditions, and the subset forwarded to the
# simulator and to stationary_moments (structural).
_MODEL_KEYS = ("omega_1", "omega_2", "noise_std", "dt", "burn_in", "n_modes")
_SIMULATOR_KEYS = ("omega_1", "omega_2", "noise_std", "dt", "burn_in")
_MOMENT_KEYS = ("omega_1", "omega_2", "noise_std", "n_modes")


def harmonic_plv(theta_1: ArrayLike, theta_2: ArrayLike, harmonic: int = 1) -> float:
    """Phase-locking value of one harmonic of the phase difference.

    Returns :math:`|\\langle e^{i h (\\theta_1 - \\theta_2)}\\rangle_t|`, the
    modulus of the sample circular moment of order :math:`h`.

    Parameters
    ----------
    theta_1, theta_2 : array_like, shape (N,)
        Phase samples of the two oscillators, in radians, on the same time grid.
    harmonic : int, default 1
        Order :math:`h` of the circular moment.

    Returns
    -------
    value : float
        Dimensionless, in [0, 1]. For ``harmonic=1`` it equals
        :func:`qot_course.quantum_ot.capstone.plv`.

    Raises
    ------
    ValueError
        If the two records do not have the same shape.

    Notes
    -----
    When to use: the second harmonic sees phase locking that the first misses,
    for example a mixture of in-phase and antiphase epochs, where the first
    moment cancels and the second does not. Moments are as in Gatto &
    Jammalamadaka (2007), DOI: 10.1016/j.stamet.2006.11.003.

    Examples
    --------
    >>> import numpy as np
    >>> ta = np.linspace(0, 20 * np.pi, 1000)
    >>> tb = ta - np.where(np.arange(1000) % 2 == 0, 0.0, np.pi)
    >>> round(harmonic_plv(ta, tb, 1), 6), round(harmonic_plv(ta, tb, 2), 6)
    (0.0, 1.0)
    """
    t1 = np.asarray(theta_1, dtype=float)
    t2 = np.asarray(theta_2, dtype=float)
    if t1.shape != t2.shape:
        raise ValueError(
            f"theta_1 and theta_2 must have the same shape, got {t1.shape} and {t2.shape}"
        )
    return plv(harmonic * t1, harmonic * t2)


def relative_phase_components(
    m1: complex, m2: complex, *, min_first_moment: float = 0.05
) -> tuple[float, float]:
    """Sine and cosine components of the relative phase of the first two moments.

    With :math:`m_1 = |m_1| e^{i\\mu_1}` and :math:`m_2 = |m_2| e^{i\\mu_2}`, the
    relative phase is :math:`\\beta = \\mu_2 - 2\\mu_1`. A shift
    :math:`\\Delta \\to \\Delta + \\phi` of the phase difference multiplies
    :math:`m_k` by :math:`e^{ik\\phi}` and leaves :math:`\\beta` unchanged. The
    function returns :math:`(|m_2| \\sin\\beta, |m_2| \\cos\\beta)`, the imaginary
    and real parts of :math:`m_2\\,\\overline{m_1}^{\\,2} / |m_1|^2`. Weighting by
    :math:`|m_2|` keeps the pair continuous where the second moment vanishes.

    Parameters
    ----------
    m1, m2 : complex
        First and second circular moments of the phase difference, sample or
        exact, with :math:`m_k = E[e^{ik\\Delta}]`. Dimensionless.
    min_first_moment : float, default 0.05
        Below this value of :math:`|m_1|` the direction :math:`\\mu_1` is not
        defined and both components are returned as NaN. The value is a
        heuristic, not derived from a test. It does not remove the sampling
        scatter of an uncoupled dyad: on 200 uncoupled records of the default
        dyad (seed 11), the sample :math:`|m_1|` fell below 0.05 in 11 % of the
        records of 200 s, 45 % of those of 800 s and 90 % of those of 3,200 s.

    Returns
    -------
    components : tuple of float
        ``(|m2| sin beta, |m2| cos beta)``, dimensionless, or ``(nan, nan)`` when
        ``abs(m1) < min_first_moment``.

    Notes
    -----
    When to use: as the classical statistic that reads the one phase of the pair
    :math:`(m_1, m_2)` that is not a mere rotation, beside the moduli
    :math:`|m_1|` and :math:`|m_2|` (the phase-locking values). It matches the
    angle returned by :func:`qot_course.quantum_ot.biharmonic.relative_phase`.
    Moments are as in Gatto & Jammalamadaka (2007),
    DOI: 10.1016/j.stamet.2006.11.003.

    Examples
    --------
    >>> import numpy as np
    >>> s, c = relative_phase_components(0.4 * np.exp(0.7j), 0.3 * np.exp(1.9j))
    >>> round(float(np.hypot(s, c)), 6), round(float(np.arctan2(s, c)), 6)
    (0.3, 0.5)
    """
    m1 = complex(m1)
    m2 = complex(m2)
    modulus_1 = abs(m1)
    if modulus_1 < min_first_moment:
        return float("nan"), float("nan")
    z = m2 * np.conj(m1) ** 2 / modulus_1**2
    return float(z.imag), float(z.real)


def _quantum_statistics(
    rho_naive: NDArray[np.complex128],
    rho_rich: NDArray[np.complex128],
    rho_weighted: Mapping[str, NDArray[np.complex128]],
) -> dict[str, float]:
    """Quantum keys of the statistics dictionary from the joint densities."""
    out = {
        "qmi_naive": coupling_qmi(rho_naive, dims=_NAIVE_DIMS),
        "qmi_rich": coupling_qmi(rho_rich, dims=_RICH_DIMS),
        "bures_rich": coupling_bures(rho_rich, dims=_RICH_DIMS),
    }
    for name, rho in rho_weighted.items():
        out[name] = coupling_qmi(rho, dims=_RICH_DIMS)
    return out


def record_statistics(
    theta_1: ArrayLike,
    theta_2: ArrayLike,
    *,
    weight_sets: Mapping[str, Sequence[float]],
    min_first_moment: float = 0.05,
) -> dict[str, float]:
    """Every statistic of the bench, computed on one pair of phase records.

    Parameters
    ----------
    theta_1, theta_2 : array_like, shape (N,)
        Phase samples of the two oscillators, in radians, on the same time grid.
    weight_sets : mapping of str to sequence of float
        One weighted embedding per entry: the key names the statistic and the
        value is the weight triple ``(w_DC, w_1, w_2)`` passed to
        :func:`qot_course.quantum_ot.embeddings.weighted_multifreq_state` with
        harmonics ``(1, 2)``.
    min_first_moment : float, default 0.05
        Threshold on the sample :math:`|m_1|` below which the relative-phase
        components are NaN; see :func:`relative_phase_components` (heuristic).

    Returns
    -------
    stats : dict of str to float
        Keys :data:`STATISTIC_KEYS_BASE` followed by the keys of ``weight_sets``.
        ``plv`` and ``plv2`` are the moduli of the sample moments of order one and
        two; ``relphase_sin`` and ``relphase_cos`` the relative-phase components;
        ``qmi_naive`` the quantum mutual information (nats) of
        :func:`qot_course.quantum_ot.capstone.joint_density_matrix`; ``qmi_rich``
        and ``bures_rich`` the quantum mutual information (nats) and Bures
        distance of the equal-weight qutrit embedding with harmonics ``(1, 2)``;
        each weighted key the quantum mutual information (nats) of its weighted
        qutrit embedding.

    Raises
    ------
    ValueError
        If the two records do not have the same shape.

    Notes
    -----
    When to use: on each simulated record of a set, to build the distributions
    that the detection and reliability comparisons of the bench summarise. Its
    exact long-run counterpart is :func:`long_run_statistics`. Moments are as in
    Gatto & Jammalamadaka (2007), DOI: 10.1016/j.stamet.2006.11.003.

    Examples
    --------
    >>> from qot_course.quantum_ot.biharmonic import simulate_biharmonic_dyad
    >>> t1, t2 = simulate_biharmonic_dyad(0.1, 0.0, duration=200.0, seed=0)
    >>> stats = record_statistics(t1[0], t2[0], weight_sets={"qmi_w": (1.0, 0.2, 2.0)})
    >>> list(stats)[-1]
    'qmi_w'
    """
    t1 = np.asarray(theta_1, dtype=float).ravel()
    t2 = np.asarray(theta_2, dtype=float).ravel()
    if t1.shape != t2.shape:
        raise ValueError(
            f"theta_1 and theta_2 must have the same shape, got {t1.shape} and {t2.shape}"
        )
    delta = t1 - t2
    m1 = complex(np.mean(np.exp(1j * delta)))
    m2 = complex(np.mean(np.exp(2j * delta)))
    rel_sin, rel_cos = relative_phase_components(
        m1, m2, min_first_moment=min_first_moment
    )
    rho_naive = joint_density_matrix(t1, t2)
    rho_rich = joint_density_from_states(
        multifreq_state(t1, harmonics=_RICH_HARMONICS),
        multifreq_state(t2, harmonics=_RICH_HARMONICS),
    )
    rho_weighted = {
        name: joint_density_from_states(
            weighted_multifreq_state(t1, harmonics=_RICH_HARMONICS, weights=w),
            weighted_multifreq_state(t2, harmonics=_RICH_HARMONICS, weights=w),
        )
        for name, w in weight_sets.items()
    }
    return {
        "plv": harmonic_plv(t1, t2, 1),
        "plv2": harmonic_plv(t1, t2, 2),
        "relphase_sin": rel_sin,
        "relphase_cos": rel_cos,
        **_quantum_statistics(rho_naive, rho_rich, rho_weighted),
    }


def long_run_statistics(
    moments: NDArray[np.complex128],
    *,
    weight_sets: Mapping[str, Sequence[float]],
    min_first_moment: float = 0.05,
) -> dict[str, float]:
    """Every statistic of the bench, computed exactly from stationary moments.

    Parameters
    ----------
    moments : ndarray of complex, shape (K,) with K >= 3
        Circular moments of the phase difference, entry ``k`` equal to
        :math:`E[e^{ik\\Delta}]` with :math:`\\Delta = \\theta_1 - \\theta_2`, as
        returned by :func:`qot_course.quantum_ot.biharmonic.stationary_moments`.
    weight_sets : mapping of str to sequence of float
        As in :func:`record_statistics`.
    min_first_moment : float, default 0.05
        As in :func:`record_statistics`; the same rule applies to the exact
        :math:`|m_1|` (heuristic).

    Returns
    -------
    stats : dict of str to float
        The keys of :func:`record_statistics`, in the same order. ``plv`` and
        ``plv2`` are :math:`|m_1|` and :math:`|m_2|`; the quantum keys are
        computed on the densities of
        :func:`qot_course.quantum_ot.biharmonic.long_run_joint_density` with the
        same harmonics, weights and subsystem dimensions as on a record.

    Raises
    ------
    ValueError
        If ``moments`` holds fewer than three entries.

    Notes
    -----
    When to use: to put the value a statistic tends to on long records beside
    its distribution over finite records, which separates sampling scatter from
    a genuine difference between two settings. The values are long-run limits:
    they hold under the assumptions of
    :func:`qot_course.quantum_ot.biharmonic.long_run_joint_density`, and a
    finite record also carries a sampling bias, positive for every modulus and
    every information measure. Moments are as in Gatto & Jammalamadaka (2007),
    DOI: 10.1016/j.stamet.2006.11.003.

    Examples
    --------
    >>> from qot_course.quantum_ot.biharmonic import stationary_moments
    >>> stats = long_run_statistics(stationary_moments(0.0, 0.0), weight_sets={})
    >>> bool(abs(stats["plv"]) < 1e-9 and abs(stats["qmi_rich"]) < 1e-9)
    True
    """
    m = np.asarray(moments, dtype=complex)
    if m.ndim != 1 or m.shape[0] < 3:
        raise ValueError(
            f"moments must be a 1-D array of at least 3 entries, got shape {m.shape}"
        )
    rel_sin, rel_cos = relative_phase_components(
        m[1], m[2], min_first_moment=min_first_moment
    )
    rho_naive = long_run_joint_density(m, harmonics=_NAIVE_HARMONICS)
    rho_rich = long_run_joint_density(m, harmonics=_RICH_HARMONICS)
    rho_weighted = {
        name: long_run_joint_density(m, harmonics=_RICH_HARMONICS, weights=w)
        for name, w in weight_sets.items()
    }
    return {
        "plv": float(abs(m[1])),
        "plv2": float(abs(m[2])),
        "relphase_sin": rel_sin,
        "relphase_cos": rel_cos,
        **_quantum_statistics(rho_naive, rho_rich, rho_weighted),
    }


def _finite(values: ArrayLike) -> NDArray[np.float64]:
    """Flattened float copy of ``values`` with every NaN and infinity removed."""
    x = np.asarray(values, dtype=float).ravel()
    return x[np.isfinite(x)]


def _mean_and_sd(values: ArrayLike) -> tuple[float, float]:
    """Mean and standard deviation (ddof=1) of the finite values, NaN below two."""
    x = _finite(values)
    if x.size < _MIN_FINITE_VALUES:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1))


def signed_score(observed: ArrayLike, null: ArrayLike) -> float:
    """Mean observed difference in units of the scatter of null differences.

    Returns :math:`\\bar d / s_0`, where :math:`\\bar d` is the mean of the
    finite observed differences and :math:`s_0` the standard deviation
    (``ddof=1``) of the finite null differences. The sign of the observed
    differences is kept, and no absolute value is taken on either side.

    Parameters
    ----------
    observed : array_like, shape (N,)
        Differences between two settings, record by record (for example B - A),
        in the units of the statistic. NaN entries are ignored.
    null : array_like, shape (M,)
        Differences between two independent sets of the same setting (for
        example A' - A''), in the same units. NaN entries are ignored.

    Returns
    -------
    score : float
        Dimensionless. NaN when either side has fewer than two finite values,
        or when the finite null values are all equal (zero scatter), so that a
        constant statistic never produces an infinite score.

    Notes
    -----
    When to use: to say how far, and in which direction, a statistic moves
    between two settings compared with how much it moves between two sets of
    the same setting. Why the sign is kept: an earlier analysis in this course
    compared an observed difference with the absolute values of a small set of
    null differences. Taking the absolute value folds the null onto the
    positive half-line, which shrinks its apparent scatter around zero and
    inflates the score; it also hides a statistic that moves in the wrong
    direction. Here both sides stay signed, and a negative score means the
    statistic decreased from the first setting to the second.

    The score assumes nothing about the shape of the null and is not
    invariant under a monotone change of the statistic; :func:`auc` is, which
    is why the protocol reports both.

    Examples
    --------
    >>> import numpy as np
    >>> round(signed_score([-2.0, -2.0], [-1.0, 1.0, -1.0, 1.0]), 4)
    -1.7321
    >>> bool(np.isnan(signed_score([1.0, 2.0], [3.0, 3.0, 3.0])))
    True
    """
    obs = _finite(observed)
    nul = _finite(null)
    if obs.size < _MIN_FINITE_VALUES or nul.size < _MIN_FINITE_VALUES:
        return float("nan")
    # Test constancy exactly rather than sd == 0: the mean of equal values that
    # are not exactly representable can leave a round-off scatter near 1e-17,
    # which would turn a constant statistic into an enormous score.
    if np.max(nul) == np.min(nul):
        return float("nan")
    return float(np.mean(obs) / np.std(nul, ddof=1))


def auc(a: ArrayLike, b: ArrayLike) -> float:
    """Area under the ROC curve that separates the values of ``a`` from ``b``.

    Returns the probability that a value drawn from ``b`` exceeds a value drawn
    from ``a``, over all pairs of finite values, with ties counted one half.
    This is the Mann--Whitney statistic of ``b`` divided by the number of pairs,
    which Hanley & McNeil (1982) showed equals the area under the empirical ROC
    curve.

    Parameters
    ----------
    a, b : array_like, shape (N,) and (M,)
        Values of one statistic on the records of two settings, in the units of
        the statistic. NaN entries are ignored.

    Returns
    -------
    area : float
        Dimensionless, in [0, 1]: 0.5 when the two sets are indistinguishable,
        1 when every value of ``b`` exceeds every value of ``a``, 0 in the
        reverse case. NaN when either side has no finite value.

    Notes
    -----
    When to use: beside :func:`signed_score`. The area depends only on the
    ranks of the values, so it is unchanged by any strictly increasing map of
    the statistic and is not dominated by a few extreme records. The score is
    in units of a null scatter and changes when the statistic is reshaped, for
    example when an information measure is compared with its square root;
    reporting both separates a real difference from an effect of scale.
    Reference: J. A. Hanley, B. J. McNeil, *Radiology* **143**, 29 (1982),
    DOI: 10.1148/radiology.143.1.7063747.

    Examples
    --------
    >>> round(auc([0.1, 0.2, 0.3], [0.25, 0.4, 0.5]), 4)
    0.8889
    >>> auc([0.1, 0.2], [0.1, 0.2])
    0.5
    """
    xa = _finite(a)
    xb = _finite(b)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    # Mann-Whitney U of b from the joint mid-ranks: the sum of the ranks of b
    # minus its smallest possible value counts the pairs with b > a, ties one
    # half. Computed directly so that no p-value (and no warning on a set of
    # identical values) is involved.
    ranks = rankdata(np.concatenate([xa, xb]))
    u_b = float(np.sum(ranks[xa.size :])) - xb.size * (xb.size + 1) / 2.0
    return u_b / (xa.size * xb.size)


def detection_rate(
    observed: ArrayLike, null: ArrayLike, *, false_alarm_rate: float = 0.05
) -> float:
    """Fraction of observed differences outside the central band of the null.

    The band runs from the ``false_alarm_rate / 2`` quantile to the
    ``1 - false_alarm_rate / 2`` quantile of the finite null differences, so
    both tails count and a difference of either sign can be detected.

    Parameters
    ----------
    observed : array_like, shape (N,)
        Differences between two settings, record by record, in the units of the
        statistic. NaN entries are ignored.
    null : array_like, shape (M,)
        Differences between two independent sets of the same setting, in the
        same units. NaN entries are ignored.
    false_alarm_rate : float, default 0.05
        Fraction of null differences expected outside the band, in (0, 1).

    Returns
    -------
    rate : float
        Fraction in [0, 1] of the finite observed differences strictly below
        the lower or strictly above the upper bound. NaN when either side has
        fewer than two finite values, or when the finite null values are all
        equal (a band of zero width), as in :func:`signed_score`.

    Raises
    ------
    ValueError
        If ``false_alarm_rate`` is not strictly between 0 and 1.

    Notes
    -----
    When to use: to read, record by record, how often a statistic would flag
    the difference between two settings at a fixed false-alarm rate. When the
    observed and null differences have the same distribution the rate is close
    to ``false_alarm_rate``; with few null values the band is coarse and the
    false-alarm rate is only approximate.

    The band comes from the scatter of the null alone. In the bench the null
    is made of two sets of the reference condition A, so the rate rises above
    ``false_alarm_rate`` when the compared condition is merely more variable
    than A, with no shift of the mean. A high rate together with a
    :func:`signed_score` near zero and an :func:`auc` near 0.5 reads "the
    compared condition is more variable", not "its mean differs".

    Examples
    --------
    >>> import numpy as np
    >>> null = np.random.default_rng(0).standard_normal(10_000)
    >>> detection_rate(null[:100] + 10.0, null)
    1.0
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError(
            f"false_alarm_rate must be strictly between 0 and 1, got {false_alarm_rate}"
        )
    obs = _finite(observed)
    nul = _finite(null)
    if obs.size < _MIN_FINITE_VALUES or nul.size < _MIN_FINITE_VALUES:
        return float("nan")
    # A constant null gives a band of zero width, which would flag every value
    # that differs from it; treat it as undefined, as signed_score does.
    if np.max(nul) == np.min(nul):
        return float("nan")
    low, high = np.quantile(nul, [false_alarm_rate / 2.0, 1.0 - false_alarm_rate / 2.0])
    return float(np.mean((obs < low) | (obs > high)))


def _set_statistics(
    condition: tuple[float, float],
    *,
    label: str,
    seed: np.random.SeedSequence,
    duration: float,
    n_records: int,
    simulator_kwargs: Mapping[str, float],
    weight_sets: Mapping[str, Sequence[float]],
    min_first_moment: float,
) -> dict[str, NDArray[np.float64]]:
    """Simulate one record set and reduce it to one array of values per statistic.

    The phases are local to this call, so they are released before the next set
    is simulated and two sets of phases are never held at once.
    """
    k1, k2 = condition
    logger.info(
        "Simulating set %s: K1=%g, K2=%g, %d records of %g s",
        label,
        k1,
        k2,
        n_records,
        duration,
    )
    theta_1, theta_2 = simulate_biharmonic_dyad(
        k1,
        k2,
        n_records=n_records,
        duration=duration,
        seed=seed,
        **simulator_kwargs,
    )
    per_record = [
        record_statistics(
            theta_1[r],
            theta_2[r],
            weight_sets=weight_sets,
            min_first_moment=min_first_moment,
        )
        for r in range(n_records)
    ]
    return {key: np.array([s[key] for s in per_record]) for key in per_record[0]}


def compare_conditions(
    condition_a: tuple[float, float],
    condition_b: tuple[float, float],
    *,
    duration: float,
    n_records: int,
    seed: int | np.random.SeedSequence,
    model: Mapping[str, float],
    weight_sets: Mapping[str, Sequence[float]],
    false_alarm_rate: float = 0.05,
    min_first_moment: float = 0.05,
) -> list[dict[str, float | str]]:
    """Compare every statistic of the bench between two coupling settings.

    Six independent record sets are drawn from
    ``np.random.SeedSequence(seed).spawn(6)`` in the fixed order A, B, A', A'',
    B', B'', where A and its copies use ``condition_a`` and B and its copies use
    ``condition_b``. The observed differences are B - A, record by record. The
    null differences are A' - A'', two sets of the reference setting; B' - B''
    gives the scatter of the second setting for comparison. Each set is
    simulated, reduced to its statistics and released before the next one, so
    memory holds one set of phases at a time.

    Parameters
    ----------
    condition_a, condition_b : tuple of float
        Couplings ``(K1, K2)`` in rad/s of the reference setting A and of the
        compared setting B.
    duration : float
        Analysed length of each record in seconds, after the discarded burn-in.
    n_records : int
        Number of records in each of the six sets.
    seed : int or numpy.random.SeedSequence
        Master seed; the six sets use its six spawned children. A
        ``SeedSequence`` is copied before spawning, so the caller's object is
        left unchanged and two calls with it give the same rows; an integer
        gives the same rows as ``np.random.SeedSequence`` of that integer.
    model : mapping of str to float
        Model settings with the keys ``omega_1``, ``omega_2`` (rad/s),
        ``noise_std`` (rad/sqrt(s)), ``dt`` and ``burn_in`` (s), forwarded to
        :func:`qot_course.quantum_ot.biharmonic.simulate_biharmonic_dyad`, and
        ``n_modes``, forwarded with the frequencies and the noise to
        :func:`qot_course.quantum_ot.biharmonic.stationary_moments`.
    weight_sets : mapping of str to sequence of float
        Weighted embeddings, as in :func:`record_statistics`.
    false_alarm_rate : float, default 0.05
        Two-sided false-alarm rate of :func:`detection_rate`.
    min_first_moment : float, default 0.05
        Threshold on :math:`|m_1|` of the relative-phase components, as in
        :func:`record_statistics` (heuristic).

    Returns
    -------
    rows : list of dict
        One row per statistic, in the order of :func:`record_statistics`, with
        the keys ``statistic`` (name), ``mean_a``, ``mean_b``, ``sd_a``,
        ``sd_b`` (mean and standard deviation, ``ddof=1``, over the finite
        values of sets A and B), ``exact_a``, ``exact_b`` (long-run values of
        :func:`long_run_statistics`), ``score`` (:func:`signed_score` of B - A
        against A' - A''), ``auc`` (:func:`auc` of the raw values of A and B),
        ``detection_rate`` (:func:`detection_rate` of B - A against
        A' - A''), ``null_sd_a`` and ``null_sd_b`` (standard deviations of the
        finite values of A' - A'' and B' - B''), ``n_valid`` (number of
        records at which both A and B are finite) and ``n_null`` (number of
        records at which both A' and A'' are finite, that is, the number of
        null differences that ``score`` and ``detection_rate`` rest on).
        For the relative-phase keys ``n_null`` can be much smaller than
        ``n_records``, because a null difference needs both uncoupled records
        to pass the threshold on :math:`|m_1|`. Summaries computed on fewer
        than two finite values are NaN.

    Raises
    ------
    ValueError
        If ``model`` lacks one of its keys (the message names it), or if the
        simulator or :func:`qot_course.quantum_ot.biharmonic.stationary_moments`
        rejects a setting.

    Notes
    -----
    When to use: for every detection comparison of the bench, where a
    statistic has to tell the reference setting from a second one on records
    of finite length. A statistic can be undefined on some records: the
    relative-phase components are NaN when the sample :math:`|m_1|` is below
    ``min_first_moment``, which happens on many records of an uncoupled dyad,
    and their exact values are NaN whenever :math:`K_1 = 0`. Such records are
    left out of the summaries, ``n_valid`` reports how many remain, and
    nothing is raised. The area under the curve follows Hanley & McNeil
    (1982), DOI: 10.1148/radiology.143.1.7063747.

    The band of ``detection_rate`` comes from the scatter of condition A
    alone, so the rate rises above ``false_alarm_rate`` when B is merely more
    variable than A, with no shift of the mean. A high rate together with a
    score near zero and an area near 0.5 reads "B is more variable"; compare
    ``null_sd_b`` with ``null_sd_a`` to check it.

    Examples
    --------
    >>> model = {"omega_1": 1.0, "omega_2": 1.2, "noise_std": 0.3,
    ...          "dt": 0.05, "burn_in": 50.0, "n_modes": 200}
    >>> rows = compare_conditions((0.0, 0.0), (0.0, 0.1), duration=20.0,
    ...                           n_records=4, seed=0, model=model, weight_sets={})
    >>> rows[1]["statistic"], round(rows[1]["exact_b"], 3)
    ('plv2', 0.362)
    """
    missing = [key for key in _MODEL_KEYS if key not in model]
    if missing:
        raise ValueError(f"model is missing the key(s) {', '.join(missing)}")
    simulator_kwargs = {key: model[key] for key in _SIMULATOR_KEYS}
    moment_kwargs = {key: model[key] for key in _MOMENT_KEYS}

    exact = {
        label: long_run_statistics(
            stationary_moments(
                k1, k2, max_harmonic=_MAX_EXACT_HARMONIC, **moment_kwargs
            ),
            weight_sets=weight_sets,
            min_first_moment=min_first_moment,
        )
        for label, (k1, k2) in (("A", condition_a), ("B", condition_b))
    }

    labels = ("A", "B", "A'", "A''", "B'", "B''")
    conditions = (
        condition_a,
        condition_b,
        condition_a,
        condition_a,
        condition_b,
        condition_b,
    )
    if isinstance(seed, np.random.SeedSequence):
        # Spawning advances the child counter of the caller's object, so a second
        # call with it would draw other sets. Spawn from an equivalent copy instead.
        master = np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    else:
        master = np.random.SeedSequence(seed)
    children = master.spawn(_N_RECORD_SETS)
    sets = {
        label: _set_statistics(
            condition,
            label=label,
            seed=child,
            duration=duration,
            n_records=n_records,
            simulator_kwargs=simulator_kwargs,
            weight_sets=weight_sets,
            min_first_moment=min_first_moment,
        )
        for label, condition, child in zip(labels, conditions, children, strict=True)
    }

    rows: list[dict[str, float | str]] = []
    for key in sets["A"]:
        a, b = sets["A"][key], sets["B"][key]
        observed = b - a
        null_a = sets["A'"][key] - sets["A''"][key]
        null_b = sets["B'"][key] - sets["B''"][key]
        mean_a, sd_a = _mean_and_sd(a)
        mean_b, sd_b = _mean_and_sd(b)
        rows.append(
            {
                "statistic": key,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "sd_a": sd_a,
                "sd_b": sd_b,
                "exact_a": float(exact["A"][key]),
                "exact_b": float(exact["B"][key]),
                "score": signed_score(observed, null_a),
                "auc": auc(a, b),
                "detection_rate": detection_rate(
                    observed, null_a, false_alarm_rate=false_alarm_rate
                ),
                "null_sd_a": _mean_and_sd(null_a)[1],
                "null_sd_b": _mean_and_sd(null_b)[1],
                "n_valid": int(np.count_nonzero(np.isfinite(a) & np.isfinite(b))),
                "n_null": int(np.count_nonzero(np.isfinite(null_a))),
            }
        )
    return rows
