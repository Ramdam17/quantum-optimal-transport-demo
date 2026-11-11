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

References
----------
G. Gatto, S. R. Jammalamadaka, "The generalized von Mises distribution",
*Stat. Methodol.* **4**, 341 (2007). DOI: 10.1016/j.stamet.2006.11.003.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qot_course.quantum_ot.biharmonic import long_run_joint_density
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
