"""Two noisy phase oscillators with a free second harmonic of the coupling.

The pair of notebooks 04/11 to 04/14 couples two phase oscillators through
:math:`K \\sin(\\theta_j - \\theta_i)`, a single knob. This module adds the next
harmonic of the phase difference with its own, independent strength:

.. math::

    \\mathrm{d}\\theta_1 = [\\omega_1 + K_1 \\sin(\\theta_2 - \\theta_1)
    + K_2 \\sin 2(\\theta_2 - \\theta_1)]\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_1,

    \\mathrm{d}\\theta_2 = [\\omega_2 + K_1 \\sin(\\theta_1 - \\theta_2)
    + K_2 \\sin 2(\\theta_1 - \\theta_2)]\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_2.

For :math:`K_2 = 0` the model is exactly the Kuramoto dyad of
:func:`qot_course.quantum_ot.capstone.simulate_kuramoto_dyad`, and with the same
random stream it produces the same trajectories. For :math:`K_2 \\neq 0` the
stationary law of the phase difference :math:`\\Delta = \\theta_1 - \\theta_2` is
no longer a one-parameter family, so its first and second circular moments can
move independently.

The two-harmonic form of the phase interaction is the one used for bimanual
coordination by Haken, Kelso and Bunz (1985) and Schoener, Haken and Kelso
(1986), and for populations of coupled oscillators by Daido (1992) and Hansel,
Mato and Meunier (1993).

References
----------
H. Haken, J. A. S. Kelso, H. Bunz, "A theoretical model of phase transitions in
human hand movements", *Biol. Cybern.* **51**, 347 (1985).
DOI: 10.1007/BF00336922.

G. Schoener, H. Haken, J. A. S. Kelso, "A stochastic theory of phase transitions
in human hand movement", *Biol. Cybern.* **53**, 247 (1986).
DOI: 10.1007/BF00336995.

H. Daido, "Order function and macroscopic mutual entrainment in uniformly
coupled limit-cycle oscillators", *Prog. Theor. Phys.* **88**, 1213 (1992).
DOI: 10.1143/ptp/88.6.1213.

D. Hansel, G. Mato, C. Meunier, "Clustering and slow switching in globally
coupled phase oscillators", *Phys. Rev. E* **48**, 3470 (1993).
DOI: 10.1103/PhysRevE.48.3470.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from qot_course.quantum_ot.embeddings import weighted_multifreq_state

# Number of noise components per step: one per oscillator.
_N_OSCILLATORS = 2
# A record needs at least two samples to define a phase difference trajectory
# and an increment (structural, not a tuning choice).
_MIN_SAMPLES = 2
# The second harmonic couples Fourier modes n and n +/- 2, so the stationary
# Fokker-Planck matrix has two bands on each side of the diagonal (derived).
_BAND_HALF_WIDTH = 2
# A truncation at |n| <= n_modes is accepted when the outermost two coefficients
# on each side are below this fraction of c_0. The two outermost rows are the ones
# that lose couplings to the dropped modes, so a negligible tail there means the
# dropped modes carry negligible weight. The value 1e-10 is a heuristic chosen
# well below every tolerance the bench uses on a moment (1e-8 and above).
_CONVERGENCE_RATIO = 1e-10
# Below this modulus a sample or exact moment is treated as zero, and its phase,
# hence the relative phase, is undefined. Heuristic: far above double-precision
# round-off on moments of order one (about 1e-16), far below any moment the bench
# reads as nonzero.
_MIN_MODULUS = 1e-12


def simulate_biharmonic_dyad(
    k1: float,
    k2: float,
    *,
    n_records: int = 1,
    duration: float = 200.0,
    dt: float = 0.05,
    omega_1: float = 1.0,
    omega_2: float = 1.2,
    noise_std: float = 0.3,
    burn_in: float = 50.0,
    random_initial_phases: bool = True,
    seed: int | np.random.SeedSequence = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler--Maruyama simulation of a noisy dyad with a free second harmonic.

    Integrates the two-oscillator model of the module docstring for ``n_records``
    independent records at once (vectorised over records, looping over time).

    Parameters
    ----------
    k1 : float
        First-harmonic coupling :math:`K_1` (rad/s).
    k2 : float
        Second-harmonic coupling :math:`K_2` (rad/s); the sign is meaningful.
    n_records : int, default 1
        Number of independent records, each with its own noise and start.
    duration : float, default 200.0
        Analysed length of each record after the burn-in (s).
    dt : float, default 0.05
        Euler--Maruyama step (s).
    omega_1, omega_2 : float, default 1.0 and 1.2
        Natural frequencies (rad/s), as in notebook 04/11.
    noise_std : float, default 0.3
        Noise amplitude :math:`\\sigma` (rad / sqrt(s)).
    burn_in : float, default 50.0
        Time simulated and discarded before the analysed record (s).
    random_initial_phases : bool, default True
        Draw the two initial phases uniformly on :math:`[0, 2\\pi)` for every
        record; if False both start at zero, as in the course's simulator.
    seed : int or numpy.random.SeedSequence, default 0
        Seed of the random generator.

    Returns
    -------
    theta_1, theta_2 : ndarray of float, shape (n_records, int(duration / dt))
        Unwrapped phases in radians. Sample ``k`` of a record is the state
        before step ``k`` after the burn-in, as in ``simulate_kuramoto_dyad``.

    Raises
    ------
    ValueError
        If ``dt``, ``duration``, ``n_records`` or ``noise_std`` is not positive,
        if ``burn_in`` is negative, or if ``duration`` gives fewer than two
        samples. The message names the offending argument.

    Notes
    -----
    When to use: to build record sets in which the second circular moment of the
    phase difference moves independently of the first. The random stream is one
    uniform draw of shape ``(2, n_records)`` for the starts (if requested), then
    one normal draw of shape ``(2, n_records)`` per step; with ``k2 = 0``,
    ``n_records = 1``, ``burn_in = 0`` and ``random_initial_phases = False`` the
    output equals that of ``simulate_kuramoto_dyad`` to rounding.

    Examples
    --------
    >>> theta_1, theta_2 = simulate_biharmonic_dyad(
    ...     0.1, 0.05, n_records=3, duration=10.0, seed=7
    ... )
    >>> theta_1.shape
    (3, 200)
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if n_records < 1:
        raise ValueError(f"n_records must be at least 1, got {n_records}")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be positive, got {noise_std}")
    if burn_in < 0.0:
        raise ValueError(f"burn_in must be non-negative, got {burn_in}")
    n_steps = int(duration / dt)
    if n_steps < _MIN_SAMPLES:
        raise ValueError(
            f"duration {duration} with dt {dt} gives {n_steps} samples; "
            f"at least {_MIN_SAMPLES} are needed"
        )
    n_burn = int(burn_in / dt)

    rng = np.random.default_rng(seed)
    if random_initial_phases:
        state = rng.uniform(0.0, 2.0 * np.pi, size=(_N_OSCILLATORS, n_records))
    else:
        state = np.zeros((_N_OSCILLATORS, n_records))

    theta_1 = np.empty((n_records, n_steps))
    theta_2 = np.empty((n_records, n_steps))
    sqrt_dt = float(np.sqrt(dt))
    # Sample k is the state before step k; the last update is never needed.
    for step in range(n_burn + n_steps):
        if step >= n_burn:
            theta_1[:, step - n_burn] = state[0]
            theta_2[:, step - n_burn] = state[1]
        if step == n_burn + n_steps - 1:
            break
        # This local difference is theta_2 - theta_1, the argument of the coupling
        # felt by oscillator 1; it is minus the phase difference Delta of the
        # module docstring and of stationary_moments.
        delta = state[1] - state[0]
        g = k1 * np.sin(delta) + k2 * np.sin(2.0 * delta)
        z = rng.standard_normal((_N_OSCILLATORS, n_records))
        state = np.stack(
            [
                state[0] + dt * (omega_1 + g) + noise_std * sqrt_dt * z[0],
                state[1] + dt * (omega_2 - g) + noise_std * sqrt_dt * z[1],
            ]
        )
    return theta_1, theta_2


def stationary_moments(
    k1: float,
    k2: float,
    *,
    omega_1: float = 1.0,
    omega_2: float = 1.2,
    noise_std: float = 0.3,
    max_harmonic: int = 2,
    n_modes: int = 200,
) -> NDArray[np.complex128]:
    """Exact long-run circular moments of the phase difference, without simulation.

    Because the coupling is antisymmetric, the phase difference
    :math:`\\Delta = \\theta_1 - \\theta_2` obeys a closed stochastic equation,

    .. math::

        \\mathrm{d}\\Delta = [\\Delta\\omega - 2K_1 \\sin\\Delta - 2K_2 \\sin 2\\Delta]
        \\,\\mathrm{d}t + \\sqrt{2D}\\,\\mathrm{d}W,
        \\qquad \\Delta\\omega = \\omega_1 - \\omega_2, \\quad D = \\sigma^2,

    where the diffusion constant is :math:`D = \\sigma^2` because the two
    independent noises add in the difference. Its stationary density
    :math:`p(\\Delta) = \\sum_n c_n e^{i n \\Delta}` solves the stationary
    Fokker--Planck equation on the circle (Risken 1996, chapter 11). On the
    Fourier coefficients this equation is a banded linear system: for every
    :math:`n \\neq 0`,

    .. math::

        (\\Delta\\omega - i D n)\\, c_n + i K_1 (c_{n-1} - c_{n+1})
        + i K_2 (c_{n-2} - c_{n+2}) = 0,

    and normalisation fixes :math:`c_0 = 1/(2\\pi)`. The system is truncated at
    :math:`|n| \\le` ``n_modes``, the row :math:`n = 0` is replaced by the
    normalisation, and the moments are :math:`m_k = E[e^{i k \\Delta}] = 2\\pi c_{-k}`.

    Parameters
    ----------
    k1 : float
        First-harmonic coupling :math:`K_1` (rad/s).
    k2 : float
        Second-harmonic coupling :math:`K_2` (rad/s); the sign is meaningful.
    omega_1, omega_2 : float, default 1.0 and 1.2
        Natural frequencies (rad/s), as in notebook 04/11.
    noise_std : float, default 0.3
        Noise amplitude :math:`\\sigma` of each oscillator (rad / sqrt(s)).
    max_harmonic : int, default 2
        Highest harmonic :math:`k` returned.
    n_modes : int, default 200
        Truncation order of the Fourier series of the density.

    Returns
    -------
    moments : ndarray of complex, shape (max_harmonic + 1,)
        Entry ``k`` is :math:`E[e^{i k \\Delta}]` with
        :math:`\\Delta = \\theta_1 - \\theta_2`; entry 0 equals 1. Dimensionless.

    Raises
    ------
    ValueError
        If ``noise_std`` is not positive, if ``max_harmonic`` is negative, if
        ``n_modes`` is too small to hold the band and the requested harmonics,
        or if the truncation has not converged (the message then names
        ``n_modes`` and asks for a larger value).

    Notes
    -----
    When to use: as the exact target of every simulated circular moment of the
    two-harmonic dyad, and to choose settings :math:`(K_1, K_2)` from their
    long-run moments without simulating. With equal natural frequencies the
    density is the generalized von Mises law, proportional to
    :math:`\\exp[(2K_1\\cos\\Delta + K_2\\cos 2\\Delta)/\\sigma^2]`
    (Gatto & Jammalamadaka 2007), which gives an independent check by
    quadrature; with :math:`K_2 = 0` as well, :math:`m_1 = I_1(\\kappa)/I_0(\\kappa)`
    with :math:`\\kappa = 2K_1/\\sigma^2`.

    The truncation is checked rather than trusted: if the two outermost
    coefficients on either side exceed ``1e-10`` times :math:`|c_0|`, a strong
    coupling has pushed weight into the dropped modes and the function raises
    instead of returning a wrong moment.

    References
    ----------
    H. Risken, *The Fokker-Planck Equation*, 2nd ed., Springer (1996), chapter 11.
    DOI: 10.1007/978-3-642-61544-3.

    R. Gatto, S. R. Jammalamadaka, "The generalized von Mises distribution",
    *Stat. Methodol.* **4**, 341 (2007). DOI: 10.1016/j.stamet.2006.11.003.

    Examples
    --------
    >>> m = stationary_moments(0.3, 0.0)
    >>> m.shape
    (3,)
    >>> round(float(abs(m[1])), 2)
    0.91
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be positive, got {noise_std}")
    if max_harmonic < 0:
        raise ValueError(f"max_harmonic must be non-negative, got {max_harmonic}")
    # The convergence check reads the two outermost modes on each side, which
    # must lie beyond both the band and the returned harmonics.
    min_modes = max(_BAND_HALF_WIDTH, max_harmonic + 2)
    if n_modes < min_modes:
        raise ValueError(
            f"n_modes must be at least {min_modes} for max_harmonic={max_harmonic}, "
            f"got {n_modes}"
        )

    detuning = omega_1 - omega_2
    diffusion = noise_std**2
    size = 2 * n_modes + 1
    centre = n_modes  # index of mode n = 0
    n = np.arange(-n_modes, n_modes + 1)

    # Banded storage for scipy.linalg.solve_banded: ab[u + i - j, j] = A[i, j]
    # with u = l = 2. Row i is the equation of mode n_i; column j is c_{n_j}.
    u = _BAND_HALF_WIDTH
    ab = np.zeros((2 * u + 1, size), dtype=np.complex128)
    ab[u] = detuning - 1j * diffusion * n  # A[i, i]
    ab[u - 1, 1:] = -1j * k1  # A[i, i + 1]: -i K1 c_{n+1}
    ab[u + 1, :-1] = 1j * k1  # A[i, i - 1]: +i K1 c_{n-1}
    ab[u - 2, 2:] = -1j * k2  # A[i, i + 2]: -i K2 c_{n+2}
    ab[u + 2, :-2] = 1j * k2  # A[i, i - 2]: +i K2 c_{n-2}
    # Replace the row n = 0 by the normalisation c_0 = 1 / (2 pi).
    ab[u, centre] = 1.0
    ab[u - 1, centre + 1] = 0.0
    ab[u + 1, centre - 1] = 0.0
    ab[u - 2, centre + 2] = 0.0
    ab[u + 2, centre - 2] = 0.0
    rhs = np.zeros(size, dtype=np.complex128)
    rhs[centre] = 1.0 / (2.0 * np.pi)
    coeffs = solve_banded((u, u), ab, rhs)

    edge = np.abs(coeffs[[0, 1, -2, -1]]).max()
    if edge > _CONVERGENCE_RATIO * abs(coeffs[centre]):
        raise ValueError(
            f"Fourier truncation has not converged with n_modes={n_modes}: the outermost "
            f"coefficients reach {edge / abs(coeffs[centre]):.2e} of c_0 (limit "
            f"{_CONVERGENCE_RATIO:.0e}); increase n_modes"
        )
    k = np.arange(max_harmonic + 1)
    return (2.0 * np.pi * coeffs[centre - k]).astype(np.complex128)


def relative_phase(moments: NDArray[np.complex128]) -> float:
    """Relative phase of the second circular moment with respect to the first.

    Returns :math:`\\beta = \\arg(m_2\\,\\overline{m_1}^{\\,2})`, the one phase of the
    pair :math:`(m_1, m_2)` that survives a constant phase shift of one
    oscillator, since such a shift multiplies :math:`m_k` by :math:`e^{ik\\phi}`.

    Parameters
    ----------
    moments : ndarray of complex, shape (K,) with K >= 3
        Circular moments with entry ``k`` equal to :math:`E[e^{ik\\Delta}]`, as
        returned by :func:`stationary_moments` or estimated from a record.

    Returns
    -------
    beta : float
        Relative phase in radians, in :math:`(-\\pi, \\pi]`; NaN when
        :math:`|m_1|` or :math:`|m_2|` is below ``1e-12``, where it is undefined.

    Raises
    ------
    ValueError
        If ``moments`` holds fewer than three entries.

    Notes
    -----
    When to use: alongside the moduli :math:`|m_1|` and :math:`|m_2|`, which
    together with :math:`\\beta` are the three real numbers on which every
    shift-invariant measure of the bench depends. Moments are as in Gatto &
    Jammalamadaka (2007), DOI: 10.1016/j.stamet.2006.11.003.

    In the two-harmonic dyad, :math:`\\beta` neither follows the sign of
    :math:`K_2` nor grows steadily with it when the natural frequencies differ.
    The values below were computed with :func:`stationary_moments` at
    :math:`K_1 = 0.10` and the default frequencies (detuning 0.2 rad/s). At
    :math:`K_2 = 0` the relative phase is already +0.249 rad, because the
    detuning skews the density. It is +0.363 rad at :math:`K_2 = +0.05` and
    -0.329 rad at :math:`K_2 = -0.05`. On a scan of :math:`K_2` over
    :math:`[-0.15, 0.15]` it changes sign only once, near :math:`K_2 = -0.016`,
    so it is negative below that value and positive above it. It increases with
    :math:`K_2` only between about -0.063 and +0.028, where it reaches its
    extremes of -0.347 and +0.411 rad. Outside that interval it falls back toward
    zero, to -0.189 rad at :math:`K_2 = -0.15` and +0.061 rad at
    :math:`K_2 = +0.15`.

    Examples
    --------
    >>> beta = relative_phase(stationary_moments(0.10, 0.05))
    >>> round(beta, 2)
    0.36
    """
    m = np.asarray(moments)
    if m.ndim != 1 or m.shape[0] < 3:
        raise ValueError(
            f"moments must be a 1-D array of at least 3 entries, got shape {m.shape}"
        )
    m1, m2 = m[1], m[2]
    if abs(m1) < _MIN_MODULUS or abs(m2) < _MIN_MODULUS:
        return float("nan")
    return float(np.angle(m2 * np.conj(m1) ** 2))


def long_run_joint_density(
    moments: NDArray[np.complex128],
    *,
    harmonics: Sequence[int] = (1, 2),
    weights: Sequence[float] | None = None,
) -> NDArray[np.complex128]:
    """Exact long-run joint density of two multi-frequency phase embeddings.

    Each oscillator is embedded as in
    :func:`qot_course.quantum_ot.embeddings.weighted_multifreq_state`, with
    channel harmonics :math:`h = (0, h_1, h_2, \\dots)` and normalised weights
    :math:`w`, and the two embeddings are combined as in
    :func:`qot_course.quantum_ot.embeddings.joint_density_from_states`, that is
    :math:`\\rho = E[|\\Psi\\rangle\\langle\\Psi|]` with
    :math:`|\\Psi\\rangle = |\\psi_A\\rangle \\otimes |\\psi_B\\rangle`. The
    element between :math:`|i\\rangle_A|j\\rangle_B` (row) and
    :math:`|k\\rangle_A|l\\rangle_B` (column) is therefore

    .. math::

        \\rho_{ij,kl} = w_i w_j w_k w_l\\,
        E\\bigl[e^{i (a\\,\\theta_1 + b\\,\\theta_2)}\\bigr],
        \\qquad a = h_i - h_k, \\quad b = h_j - h_l,

    where the row index carries the phase and the column index its conjugate.
    Writing :math:`a\\theta_1 + b\\theta_2 = (a + b) S / 2 + (a - b) \\Delta / 2`
    with :math:`S = \\theta_1 + \\theta_2` and :math:`\\Delta = \\theta_1 - \\theta_2`
    separates the sum from the difference. Under an antisymmetric coupling with
    independent noises of equal amplitude, the sum feels no coupling and is an
    unbounded diffusion with a constant drift, driven by a noise independent of
    the one that drives the difference. Over a long record the factor in
    :math:`S` therefore averages to zero whenever :math:`a + b \\neq 0`. When
    :math:`a + b = 0` the element is :math:`w_i w_j w_k w_l\\, m_a`, with
    :math:`m_a = E[e^{i a \\Delta}]` and :math:`m_{-a} = \\overline{m_a}`. The
    surviving elements thus carry the circular moments of the phase difference
    and nothing else.

    Parameters
    ----------
    moments : ndarray of complex, shape (K,)
        Circular moments of the phase difference, entry ``k`` equal to
        :math:`E[e^{ik\\Delta}]` with :math:`\\Delta = \\theta_1 - \\theta_2`, as
        returned by :func:`stationary_moments`. Entry 0 must equal 1 for the
        result to have unit trace. ``K`` must exceed the largest gap
        :math:`|h_i - h_k|`, which is ``max(harmonics)`` for positive harmonics.
        Dimensionless.
    harmonics : sequence of int, default (1, 2)
        Phase harmonics of the embedding, as in
        :func:`qot_course.quantum_ot.embeddings.weighted_multifreq_state`.
    weights : sequence of float or None, default None
        Per-channel weights ``(w_DC, w_h1, ...)``, normalised internally exactly
        as in :func:`qot_course.quantum_ot.embeddings.weighted_multifreq_state`;
        ``None`` gives equal weights.

    Returns
    -------
    rho : ndarray of complex, shape (d*d, d*d)
        Joint density with ``d = 1 + len(harmonics)``, in the Kronecker basis
        :math:`|i\\rangle_A|j\\rangle_B \\mapsto i\\,d + j` of
        :func:`qot_course.quantum_ot.embeddings.joint_density_from_states`.
        Elements whose phase does not cancel in the sum are exactly zero.

    Raises
    ------
    ValueError
        If ``moments`` is not one-dimensional or holds too few entries for the
        harmonics, or if ``weights`` has the wrong length.

    Notes
    -----
    When to use: to put an exact long-run value beside every quantum statistic
    measured on simulated records of the two-harmonic dyad, by feeding the
    result to a measure such as
    :func:`qot_course.quantum_ot.capstone.coupling_qmi`. The result is valid
    when the coupling is antisymmetric, the two noises are independent with
    equal amplitude, and the record is long enough for the sum of the phases to
    have spread over many turns; on a finite record the vanishing elements are
    only small, not zero. Moments are as in Gatto & Jammalamadaka (2007),
    DOI: 10.1016/j.stamet.2006.11.003; the diffusion of the sum follows from
    the Fokker--Planck description of Risken (1996),
    DOI: 10.1007/978-3-642-61544-3.

    Examples
    --------
    >>> rho = long_run_joint_density(np.array([1.0, 0.4 + 0.1j]), harmonics=(1,))
    >>> rho.shape
    (4, 4)
    >>> bool(np.isclose(rho[2, 1], (0.4 + 0.1j) / 4))   # |1>_A|0>_B against |0>_A|1>_B
    True
    """
    m = np.asarray(moments, dtype=complex)
    h = np.array((0, *harmonics), dtype=int)
    # Gap between channel harmonics, for system A (i, k) and for system B (j, l).
    gap = h[:, None] - h[None, :]
    max_gap = int(np.abs(gap).max())
    if m.ndim != 1 or m.shape[0] <= max_gap:
        raise ValueError(
            f"moments must be a 1-D array of at least {max_gap + 1} entries "
            f"for harmonics {tuple(harmonics)}, got shape {m.shape}"
        )
    # The embedding at theta = 0 is the normalised weight vector itself, so the
    # normalisation and the length check are exactly those of the embedding.
    embedded_zero = weighted_multifreq_state(
        np.zeros(1), harmonics=harmonics, weights=weights
    )
    w = embedded_zero[0].real
    d = h.shape[0]
    # Axes (i, j, k, l): a = h_i - h_k on A, b = h_j - h_l on B.
    a = gap[:, None, :, None]
    b = gap[None, :, None, :]
    survives = np.broadcast_to(a + b == 0, (d, d, d, d))
    a_full = np.broadcast_to(a, (d, d, d, d))
    # m_a for a >= 0 and conj(m_{-a}) for a < 0.
    moment = np.where(a_full >= 0, m[np.abs(a_full)], np.conj(m[np.abs(a_full)]))
    amplitude = np.einsum("i,j,k,l->ijkl", w, w, w, w)
    tensor = np.where(survives, amplitude * moment, 0.0)
    # Row (i, j) -> i*d + j and column (k, l) -> k*d + l: the Kronecker basis.
    return tensor.reshape(d * d, d * d)
