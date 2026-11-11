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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.optimize import brentq, least_squares

from qot_course.quantum_ot.embeddings import weighted_multifreq_state
from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)

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
# Number of evenly spaced K1 values on which |m1| - target is scanned before the
# first sign change is refined. Heuristic: on the default search interval [0, 3]
# it gives a step of 0.01 rad/s, fine enough for every |m1| curve of the bench,
# while costing a few hundred banded solves. Two crossings closer than one step
# are missed.
_K1_SCAN_POINTS = 301
# Number of starting points of the least-squares search for an equal-moduli
# partner, spread evenly in K2 over the allowed interval (plus the mirror image
# of the reference setting). Heuristic: the residual surface can have several
# local minima, and a handful of starts was enough in the trial.
_N_PAIR_STARTS = 5
# Step, cost and gradient tolerances of scipy.optimize.least_squares for the
# equal-moduli search. Heuristic: tight enough that the residual is limited by
# the reachability of the target rather than by the optimiser, so that a pair
# within 1e-6 is not reported as missed.
_PAIR_SOLVER_TOL = 1e-14
# Keywords of stationary_moments that the matched-settings solvers forward. The
# harmonic count is excluded on purpose: each solver fixes the one it needs, so a
# caller cannot make it match fewer moments than it reports (structural).
_MODEL_KEYWORDS = frozenset({"omega_1", "omega_2", "noise_std", "n_modes"})
# Extra Fourier modes, beyond twice the Carson bandwidth of the step factor, in
# the first grid tried by discretised_stationary_moments. The Bessel functions of
# the Jacobi-Anger expansion leave their oscillating range over a transition of a
# few times the cube root of their argument, which is under 10 modes for every
# step and coupling of the bench; 64 covers it several times over. Heuristic,
# and checked a posteriori on every call (see _SPECTRUM_TAIL_FRACTION).
_SPECTRUM_MARGIN = 64
# The aliasing check of discretised_stationary_moments reads the outer eighth of
# the folded spectrum on each side, indices above (1/2 - 1/8) M of a grid of M
# points. The first grid puts those indices beyond 1.5 times the bandwidth, and
# the kept band must end below them (structural, see the function's Notes).
_SPECTRUM_TAIL_FRACTION = 1.0 / 8.0
# Largest grid size tried before discretised_stationary_moments gives up on
# resolving the step factor. 2**20 points hold the spectrum of a factor whose
# bandwidth is about 260,000 modes, far beyond any setting where the scheme is
# meaningful; the cap prevents an unbounded loop and an unbounded first
# allocation, both checked before the grid is built (heuristic).
_MAX_SPECTRUM_GRID = 2**20


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


def _check_finite(**values: float) -> None:
    """Raise a ValueError naming the first argument that is NaN or infinite."""
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")


def _step_factor_band(
    k1: float,
    k2: float,
    *,
    dt: float,
    detuning: float,
    noise_std: float,
    n_modes: int,
) -> tuple[NDArray[np.complex128], int]:
    """Fourier coefficients of exp(i k dt f(Delta)) - 1 for 0 <= k <= n_modes.

    Returns the coefficient array and the half-width of the band outside which
    every coefficient, for any ``|k| <= n_modes``, is negligible against the
    size of its equation. Row ``k`` of the array holds the coefficients for
    that ``k``; column ``c`` is the coefficient of ``exp(i j Delta)`` with
    ``j`` congruent to ``c`` modulo the grid size. Row ``-k`` is the complex
    conjugate of row ``k`` with ``j -> -j``, because the factor of ``-k`` is
    the conjugate of that of ``k``, so only ``k >= 0`` is transformed. The
    grid starts from the size derived in the Notes of
    :func:`discretised_stationary_moments` and doubles until the outer part of
    the folded spectrum is negligible.
    """
    size = 2 * n_modes + 1
    k_nonneg = np.arange(n_modes + 1)
    # Carson bandwidth of the step factor: the largest derivative of its phase
    # k dt f(Delta) over the circle and over |k| <= n_modes, in modes.
    bandwidth = int(np.ceil(2.0 * n_modes * dt * (abs(k1) + 2.0 * abs(k2))))
    grid = 1 << int(np.ceil(np.log2(2 * (2 * bandwidth + _SPECTRUM_MARGIN))))
    if grid > _MAX_SPECTRUM_GRID:
        raise ValueError(
            f"the step factor of the scheme would need at least {grid} points "
            f"(Carson bandwidth {bandwidth} modes, cap {_MAX_SPECTRUM_GRID}) with "
            f"dt={dt}, k1={k1}, k2={k2}; the step is far too large for this coupling"
        )
    # Equation k of the moment system has the entries e^{-q_k} b_j^{(k)} and the
    # diagonal term expm1(-q_k), with q_k = k^2 sigma^2 dt. Its size is taken as
    # |expm1(-q_k)| + e^{-q_k} max_j |b_j^{(k)}|, which is O(dt) at a small
    # step. Every threshold below is relative to that size: the step bias lives
    # in the O(dt^2) entries of each row, and an absolute threshold would drop
    # them once dt is small enough. Row -k has the size of row k. Row 0 is
    # identically zero (its phase vanishes) and is replaced by m_0 = 1 anyway.
    damping = k_nonneg.astype(float) ** 2 * noise_std**2 * dt
    # A coefficient error below this fraction of the size of its equation,
    # summed over the at most `size` terms of the equation, stays below
    # _CONVERGENCE_RATIO times that size.
    tail_limit = _CONVERGENCE_RATIO / size
    while True:
        x = 2.0 * np.pi * np.arange(grid) / grid
        drift = detuning - 2.0 * k1 * np.sin(x) - 2.0 * k2 * np.sin(2.0 * x)
        phase = dt * k_nonneg[:, None] * drift[None, :]
        # exp(i phase) - 1 written as -2 sin^2(phase / 2) + i sin(phase): at a
        # small step the factor is 1 + O(dt), and subtracting the 1 after the
        # exponential would lose the digits that carry the drift.
        factor = -2.0 * np.sin(0.5 * phase) ** 2 + 1j * np.sin(phase)
        spectrum = np.fft.fft(factor, axis=1) / grid
        magnitude = np.abs(spectrum)
        row_size = np.abs(np.expm1(-damping)) + np.exp(-damping) * magnitude.max(axis=1)
        relative = np.divide(
            magnitude,
            row_size[:, None],
            out=np.zeros_like(magnitude),
            where=row_size[:, None] > 0.0,
        )
        folded = np.abs(np.fft.fftfreq(grid, d=1.0 / grid))
        # Largest relative modulus at each folded index |j|, over k >= 0 and,
        # through the mirror j -> -j, over k < 0.
        largest = relative.max(axis=0)
        largest = np.maximum(largest, largest[(-np.arange(grid)) % grid])
        outer = folded >= (0.5 - _SPECTRUM_TAIL_FRACTION) * grid
        tail = float(largest[outer].max())
        if tail <= tail_limit:
            break
        if 2 * grid > _MAX_SPECTRUM_GRID:
            raise ValueError(
                f"the step factor of the scheme is not resolved on {grid} points "
                f"(outer coefficients reach {tail:.2e} of their equation, limit "
                f"{tail_limit:.1e}) with dt={dt}, k1={k1}, k2={k2}; the step is "
                "far too large for this coupling"
            )
        grid *= 2
    kept = folded[largest > tail_limit]
    # No equation couples moments more than 2 n_modes apart.
    half_width = min(int(kept.max()) if kept.size else 0, 2 * n_modes)
    # Drop the negligible coefficients inside the band too, on the same
    # relative threshold. Besides matching the error bound, this keeps exact
    # zeros exact: with K1 = 0 the odd coefficients vanish in theory, and their
    # round-off, about 1e-16 of the row, would otherwise leave m1 at 1e-17.
    spectrum[relative <= tail_limit] = 0.0
    return spectrum, half_width


def discretised_stationary_moments(
    k1: float,
    k2: float,
    *,
    dt: float,
    omega_1: float = 1.0,
    omega_2: float = 1.2,
    noise_std: float = 0.3,
    max_harmonic: int = 2,
    n_modes: int = 200,
) -> NDArray[np.complex128]:
    """Exact long-run circular moments of the Euler--Maruyama scheme, unsimulated.

    :func:`simulate_biharmonic_dyad` advances both phases by one step of
    length :math:`\\delta t` with independent normal increments of variance
    :math:`\\sigma^2 \\delta t`. Subtracting the two updates as they are coded,
    the phase difference :math:`\\Delta = \\theta_1 - \\theta_2` follows the chain

    .. math::

        \\Delta' = \\Delta + \\delta t\\, f(\\Delta) + s\\,\\xi, \\qquad
        f(\\Delta) = \\Delta\\omega - 2K_1 \\sin\\Delta - 2K_2 \\sin 2\\Delta, \\qquad
        s^2 = 2\\sigma^2 \\delta t,

    with :math:`\\xi` a standard normal variable independent of :math:`\\Delta`:
    the coupling enters each oscillator once with each sign, which doubles it
    in the difference, and the two noise variances add. Because
    :math:`e^{ik\\Delta}` is :math:`2\\pi`-periodic for an integer :math:`k`, this is
    a Markov chain on the circle. At stationarity :math:`\\Delta'` has the law
    of :math:`\\Delta`, and averaging over :math:`\\xi` first gives, for every
    integer :math:`k`,

    .. math::

        m_k = e^{-k^2 s^2 / 2}\\,
        E\\bigl[e^{ik\\Delta}\\, e^{ik\\,\\delta t\\, f(\\Delta)}\\bigr]
        = e^{-k^2 \\sigma^2 \\delta t} \\sum_j a^{(k)}_j\\, m_{k+j},

    where :math:`a^{(k)}_j` are the Fourier coefficients of the periodic factor
    :math:`e^{ik\\,\\delta t\\, f(\\Delta)} = \\sum_j a^{(k)}_j e^{ij\\Delta}`.
    This is a linear system on the moments. It is truncated at
    :math:`|k| \\le` ``n_modes``, the equation :math:`k = 0` (which reads
    :math:`m_0 = m_0`) is replaced by :math:`m_0 = 1`, and the system is solved
    as a banded system (see Notes). As :math:`\\delta t \\to 0` the system
    reduces to the stationary Fokker--Planck equation solved by
    :func:`stationary_moments`.

    Parameters
    ----------
    k1 : float
        First-harmonic coupling :math:`K_1` (rad/s).
    k2 : float
        Second-harmonic coupling :math:`K_2` (rad/s); the sign is meaningful.
    dt : float
        Euler--Maruyama step :math:`\\delta t` of the simulated scheme (s).
    omega_1, omega_2 : float, default 1.0 and 1.2
        Natural frequencies (rad/s), as in notebook 04/11.
    noise_std : float, default 0.3
        Noise amplitude :math:`\\sigma` of each oscillator (rad / sqrt(s)).
    max_harmonic : int, default 2
        Highest harmonic :math:`k` returned.
    n_modes : int, default 200
        Truncation order of the moment system, :math:`|k| \\le` ``n_modes``.

    Returns
    -------
    moments : ndarray of complex, shape (max_harmonic + 1,)
        Entry ``k`` is :math:`E[e^{i k \\Delta}]` under the stationary law of the
        scheme, with :math:`\\Delta = \\theta_1 - \\theta_2`; entry 0 equals 1.
        Dimensionless. The layout is that of :func:`stationary_moments`.

    Raises
    ------
    ValueError
        If a coupling, a frequency, ``dt`` or ``noise_std`` is not finite, if
        ``dt`` or ``noise_std`` is not positive, if ``max_harmonic`` is
        negative, if ``n_modes`` is too small to hold the requested harmonics
        and the convergence check, if the truncation has not converged (the
        message then names ``n_modes`` and asks for a larger value), or if the
        step factor cannot be resolved on the largest grid allowed.

    Notes
    -----
    When to use: as the value a simulated mean converges to when the records
    are made longer and more numerous at a fixed step. :func:`stationary_moments`
    gives the long-run moments of the continuous-time model, which the
    simulator reaches only as :math:`\\delta t \\to 0`; this function gives
    those of the scheme actually simulated. Use the first to describe the
    model, and this one to judge a simulated statistic or to match two
    settings as they will be simulated. At the bench step
    :math:`\\delta t = 0.05` the two differ by a few thousandths on
    :math:`|m_2|`, and the difference shrinks in proportion to the step.

    The coefficients :math:`a^{(k)}_j` come from a fast Fourier transform of the
    factor sampled on a regular grid of :math:`M` points, minus one, written
    as :math:`-2\\sin^2(\\phi/2) + i\\sin\\phi` so that the small drift term keeps
    its digits at a small step; the rows :math:`k < 0` are the conjugates of
    the rows :math:`k > 0` with :math:`j \\to -j`.
    A closed form exists through the Jacobi--Anger expansion, a double sum of
    products of Bessel functions, but its truncation would have to be
    controlled in the same way, and the transform gives all the coefficients
    of a row at once. The same expansion says where the coefficients live:
    those of row :math:`k` are negligible beyond the Carson bandwidth
    :math:`B = 2\\,|k|\\,\\delta t\\,(|K_1| + 2|K_2|)` plus a transition of a few
    times :math:`B^{1/3}`, and they decay faster than exponentially past it.
    The transform returns each coefficient plus its aliases
    :math:`a_{j \\pm M}, a_{j \\pm 2M}, \\dots`. The first grid is the smallest
    power of two with :math:`M \\ge 2\\,(2B + 64)` at :math:`|k| = N` =
    ``n_modes``. Every threshold is relative to the size of the equation the
    coefficient enters, :math:`r_k = |e^{-q_k} - 1| + e^{-q_k} \\max_j
    |a^{(k)}_j - \\delta_{j0}|` with :math:`q_k = k^2 \\sigma^2 \\delta t`. This
    matters at a small step: each equation is of order :math:`\\delta t`, the
    step bias lives in its entries of order :math:`\\delta t^2`, and a fixed
    threshold would drop those entries once :math:`\\delta t` is small enough.
    The grid is then checked: every computed coefficient with folded index
    above :math:`3M/8` must be below :math:`\\varepsilon\\, r_k`, with
    :math:`\\varepsilon = 10^{-10} / (2N + 1)`, and the grid doubles until it
    is. Coefficients below :math:`\\varepsilon\\, r_k` are then dropped (so that
    a coefficient that vanishes in theory, such as every odd one when
    :math:`K_1 = 0`, is exactly zero: its round-off is about :math:`10^{-16}
    r_k`), which leaves a banded system of half-width :math:`W < 3M/8`, solved
    with :func:`scipy.linalg.solve_banded`. The aliases that reach a kept
    coefficient come from indices above :math:`M - W > 5M/8`, past the checked
    region, where the coefficients are smaller still. Each equation is a sum
    of at most :math:`2N + 1` coefficients times moments of modulus at most
    one, so the dropped and aliased parts change it by less than
    :math:`10^{-10} r_k`. This bounds the residual of each equation relative to
    its own size, not the error of the moments, which also depends on the
    conditioning of the system. That error was measured instead, against a
    dense solve of the same system on 4096 points that drops nothing: it is
    below :math:`10^{-13}` on the settings of the bench for every step from
    :math:`10^{-6}` to :math:`0.5`. At the bench step the band is a few tens
    of modes wide and one call takes about 2 ms.

    The truncation is checked as in :func:`stationary_moments`: if the two
    outermost moments on either side exceed ``1e-10`` (the modulus of
    :math:`m_0`), the function raises. The default ``n_modes = 200`` passes
    this check, and doubling it changes :math:`m_1` and :math:`m_2` by less than
    ``1e-10``, on the settings of the bench at :math:`\\delta t = 0.05`,
    including the slowly decaying trap setting :math:`(K_1, K_2) = (0, 0.3)`,
    and at :math:`\\delta t = 10^{-4}` and :math:`10^{-6}`.

    The bench reads every statistic from these moments through
    :func:`long_run_joint_density`, which needs two properties of the
    stationary law: each phase is uniform, and only the difference carries
    structure. Both hold for the scheme. Adding the two coded updates, the sum
    :math:`S = \\theta_1 + \\theta_2` receives :math:`\\delta t(\\omega_1 + \\omega_2)`,
    in which the coupling cancels, and the noise :math:`\\sigma\\sqrt{\\delta t}
    (z_1 + z_2)`. The pair :math:`z_1 + z_2, z_1 - z_2` is jointly normal with
    zero covariance, so the noise of the sum is independent of the chain of
    the difference, and after :math:`n` steps the factor
    :math:`E[e^{i c S / 2}]` with :math:`c \\neq 0` is damped by
    :math:`e^{-c^2 \\sigma^2 \\delta t\\, n / 4}` whatever the start. The joint
    density therefore keeps only the elements carried by :math:`m_k`, as in
    continuous time.

    What is established: the recurrence is exact for the update as coded,
    with no approximation in :math:`\\delta t`, and the two numerical
    truncations are checked on every call. What is not: the existence and
    uniqueness of the stationary law of the chain is assumed here from the
    positive noise on a compact state space, not proved, and a singular
    system would surface as a :class:`numpy.linalg.LinAlgError` raised by
    :func:`scipy.linalg.solve_banded`. The
    derivation is elementary, so no source is cited.

    Examples
    --------
    >>> m = discretised_stationary_moments(0.3, 0.0, dt=0.05)
    >>> m.shape
    (3,)
    >>> round(float(abs(m[2]) - abs(stationary_moments(0.3, 0.0)[2])), 5)
    -0.00356
    """
    _check_finite(
        k1=k1, k2=k2, dt=dt, omega_1=omega_1, omega_2=omega_2, noise_std=noise_std
    )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be positive, got {noise_std}")
    if max_harmonic < 0:
        raise ValueError(f"max_harmonic must be non-negative, got {max_harmonic}")
    # The convergence check reads the two outermost moments on each side, which
    # must lie beyond the returned harmonics.
    min_modes = max_harmonic + 2
    if n_modes < min_modes:
        raise ValueError(
            f"n_modes must be at least {min_modes} for max_harmonic={max_harmonic}, "
            f"got {n_modes}"
        )

    size = 2 * n_modes + 1
    centre = n_modes  # index of the moment m_0
    ks = np.arange(-n_modes, n_modes + 1)
    spectrum, width = _step_factor_band(
        k1,
        k2,
        dt=dt,
        detuning=omega_1 - omega_2,
        noise_std=noise_std,
        n_modes=n_modes,
    )
    grid = spectrum.shape[1]
    # Equation of row i (harmonic k = ks[i]):
    #   e^{-q_k} sum_j b^{(k)}_j m_{k+j} + (e^{-q_k} - 1) m_k = 0,
    # with b = a - delta_{j0} and q_k = k^2 sigma^2 dt. Column c holds m_{ks[c]},
    # so the coefficient index is j = c - i. Banded storage of
    # scipy.linalg.solve_banded: ab[width + i - c, c] = A[i, c] for |c - i| <= width.
    damping = ks.astype(float) ** 2 * noise_std**2 * dt
    offsets = np.arange(-width, width + 1)  # j = c - i
    cols = np.arange(size)
    rows = cols[None, :] - offsets[:, None]  # i = c - j, shape (2 width + 1, size)
    valid = (rows >= 0) & (rows < size)
    safe_rows = np.where(valid, rows, centre)
    # b^{(k)}_j for k >= 0 from the transformed rows, and for k < 0 as the
    # conjugate of b^{(-k)}_{-j}.
    k_of_row = ks[safe_rows]
    j = np.broadcast_to(offsets[:, None], k_of_row.shape)
    coefficient = np.where(
        k_of_row >= 0,
        spectrum[np.abs(k_of_row), j % grid],
        np.conj(spectrum[np.abs(k_of_row), (-j) % grid]),
    )
    ab = np.where(valid, np.exp(-damping)[safe_rows] * coefficient, 0.0).astype(
        np.complex128
    )
    # Row index width - j of ab holds offset j, so offset j sits at width - j.
    ab = ab[::-1]
    ab[width] += np.expm1(-damping)
    # Replace the trivial equation k = 0 by the normalisation m_0 = 1.
    inside = np.abs(offsets) <= n_modes  # columns centre + j that exist
    ab[width - offsets[inside], centre + offsets[inside]] = 0.0
    ab[width, centre] = 1.0
    rhs = np.zeros(size, dtype=np.complex128)
    rhs[centre] = 1.0
    moments = solve_banded((width, width), ab, rhs)

    edge = float(np.abs(moments[[0, 1, -2, -1]]).max())
    if edge > _CONVERGENCE_RATIO:
        raise ValueError(
            f"moment truncation has not converged with n_modes={n_modes}: the "
            f"outermost moments reach {edge:.2e} (limit {_CONVERGENCE_RATIO:.0e}); "
            "increase n_modes"
        )
    k = np.arange(max_harmonic + 1)
    return moments[centre + k].astype(np.complex128)


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
    only small, not zero. The surviving elements match those of a record only
    once the phase difference has explored its stationary law within that
    record. A strong second harmonic prevents this on records of realistic
    length: near :math:`K_2 = 0.3` the difference stays in one of its two wells
    for a long time, and a record then carries a first moment far from the
    stationary one. Moments are as in Gatto & Jammalamadaka (2007),
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


def _check_model_keywords(model: dict[str, float]) -> None:
    """Reject any model keyword the matched-settings solvers do not forward.

    Parameters
    ----------
    model : dict of str to float
        Keyword arguments meant for :func:`stationary_moments`.

    Raises
    ------
    ValueError
        If a keyword is not one of ``omega_1``, ``omega_2``, ``noise_std`` or
        ``n_modes``; the message names it.
    """
    unknown = sorted(set(model) - _MODEL_KEYWORDS)
    if unknown:
        raise ValueError(
            f"unsupported model keyword(s) {unknown}; allowed: {sorted(_MODEL_KEYWORDS)}"
        )


def _check_matching_step(dt: float | None) -> None:
    """Reject a step that is given but not a finite positive number.

    Parameters
    ----------
    dt : float or None
        Step of the scheme the solvers match on, or None for continuous time.

    Raises
    ------
    ValueError
        If ``dt`` is not None and is not finite and positive; the message names ``dt``.
    """
    if dt is not None and not (np.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be None or a finite positive step, got {dt}")


def _matched_moments(
    k1: float,
    k2: float,
    *,
    max_harmonic: int,
    dt: float | None,
    model: dict[str, float],
) -> NDArray[np.complex128]:
    """Moments the matched-settings solvers compare: continuous time or the scheme.

    With ``dt`` None they come from :func:`stationary_moments`, otherwise from
    :func:`discretised_stationary_moments` at that step, with the same model
    keywords in both cases.
    """
    if dt is None:
        return stationary_moments(k1, k2, max_harmonic=max_harmonic, **model)
    return discretised_stationary_moments(
        k1, k2, dt=dt, max_harmonic=max_harmonic, **model
    )


def solve_k1_for_plv(
    target: float,
    k2: float,
    *,
    k1_search_max: float = 3.0,
    dt: float | None = None,
    **model: float,
) -> float:
    """First-harmonic coupling that gives a requested phase-locking value.

    Finds the first :math:`K_1 \\ge 0`, on the scan described below, at which the
    exact long-run phase-locking value :math:`|m_1| = |E[e^{i\\Delta}]|` equals
    ``target``, with :math:`K_2` held at ``k2``. With ``dt`` None the matched
    law is the stationary law of the continuous-time model
    (:func:`stationary_moments`); with a step ``dt`` it is the stationary law
    of the Euler--Maruyama scheme simulated at that step
    (:func:`discretised_stationary_moments`). The function
    :math:`|m_1| - \\text{target}` is evaluated on 301 evenly spaced values of
    :math:`K_1` in ``[0, k1_search_max]`` (a step of
    0.01 rad/s on the default interval), and the first sign change is refined
    with Brent's method.

    Parameters
    ----------
    target : float
        Requested phase-locking value :math:`|m_1|`, dimensionless.
    k2 : float
        Second-harmonic coupling :math:`K_2` (rad/s), held fixed.
    k1_search_max : float, default 3.0
        Upper end of the scanned interval of :math:`K_1` (rad/s).
    dt : float or None, default None
        Step of the simulated scheme to match on (s), or None to match the
        continuous-time model.
    **model : float
        Forwarded to :func:`stationary_moments` or
        :func:`discretised_stationary_moments`: ``omega_1``, ``omega_2``,
        ``noise_std`` and ``n_modes``.

    Returns
    -------
    k1 : float
        First-harmonic coupling (rad/s) at which :math:`|m_1|` equals
        ``target`` to the precision of Brent's method.

    Raises
    ------
    ValueError
        If a keyword of ``model`` is not one of the four listed above (the
        message names it), if ``dt`` is given but not finite and positive, if
        ``k1_search_max`` is not positive, or if no sign
        change of :math:`|m_1| - \\text{target}` is found on the scan. The last
        message names ``target`` and ``k2``.

    Notes
    -----
    When to use: to build two settings with the same phase-locking value and
    different second-harmonic couplings, so that any difference a statistic
    sees between them cannot come from :math:`|m_1|`. Match on the scheme
    (``dt`` set to the simulation step) when the two settings will be
    simulated and compared, because the step shifts :math:`|m_1|` by a
    different amount in each setting.

    The root found is the first crossing on the scan from zero, not a
    guaranteed global smallest root. With the default detuning, :math:`|m_1|`
    need not grow monotonically with :math:`K_1` at fixed :math:`K_2`, so the
    curve may cross ``target`` more than once. Two crossings that both fall
    between the same pair of neighbouring grid points cancel in the sign test
    and are missed.

    Examples
    --------
    >>> k1 = solve_k1_for_plv(0.3, 0.0)
    >>> round(float(abs(stationary_moments(k1, 0.0)[1])), 6)
    0.3
    """
    _check_model_keywords(model)
    _check_matching_step(dt)
    if k1_search_max <= 0.0:
        raise ValueError(f"k1_search_max must be positive, got {k1_search_max}")

    def gap(k1: float) -> float:
        moments = _matched_moments(k1, k2, max_harmonic=1, dt=dt, model=model)
        return float(abs(moments[1])) - target

    grid = np.linspace(0.0, k1_search_max, _K1_SCAN_POINTS)
    values = np.array([gap(k1) for k1 in grid])
    for i in range(grid.size - 1):
        if values[i] == 0.0:
            return float(grid[i])
        if values[i] * values[i + 1] < 0.0:
            return float(brentq(gap, grid[i], grid[i + 1]))
    if values[-1] == 0.0:
        return float(grid[-1])
    raise ValueError(
        f"no k1 in [0, {k1_search_max}] gives |m1| = target={target} at k2={k2}"
    )


@dataclass(frozen=True)
class EqualModuliPair:
    """A reference setting and a partner with the same two moment moduli.

    Attributes
    ----------
    k1_ref, k2_ref : float
        Reference couplings :math:`(K_1, K_2)` (rad/s).
    k1, k2 : float
        Partner couplings (rad/s); ``k2`` lies between 0 and the bound on the
        side opposite to ``k2_ref``, and is strictly nonzero when ``found``.
    residual : float
        Larger of :math:`\\bigl||m_1| - |m_1^{\\mathrm{ref}}|\\bigr|` and
        :math:`\\bigl||m_2| - |m_2^{\\mathrm{ref}}|\\bigr|` at the partner,
        dimensionless.
    found : bool
        True when ``residual`` is within the tolerance asked for and ``k2`` is
        strictly of the sign opposite to ``k2_ref``; when False the partner is
        only the closest setting found, and a comparison that assumes equal
        moduli must not be made with it.
    """

    k1_ref: float
    k2_ref: float
    k1: float
    k2: float
    residual: float
    found: bool


def solve_equal_moduli_pair(
    k1_ref: float,
    k2_ref: float,
    *,
    k2_bound: float = 0.15,
    tolerance: float = 1e-6,
    k1_search_max: float = 3.0,
    dt: float | None = None,
    **model: float,
) -> EqualModuliPair:
    """Partner setting with the same moduli of both moments and opposite K2.

    Searches for :math:`(K_1, K_2)` with :math:`K_2` of the sign opposite to
    ``k2_ref`` and :math:`|K_2| \\le` ``k2_bound`` such that :math:`|m_1|` and
    :math:`|m_2|` equal those of the reference. With ``dt`` None the moments
    are those of the continuous-time model (:func:`stationary_moments`); with
    a step ``dt`` they are those of the Euler--Maruyama scheme simulated at
    that step (:func:`discretised_stationary_moments`), for the reference and
    the partner alike.
    The two modulus differences are minimised with
    :func:`scipy.optimize.least_squares` under the bounds
    :math:`K_1 \\in [0,` ``k1_search_max`` :math:`]` and :math:`K_2` between
    :math:`-\\mathrm{sign}(K_2^{\\mathrm{ref}})` ``k2_bound`` and 0, from six
    starting points: the mirror image :math:`(K_1^{\\mathrm{ref}}, -K_2^{\\mathrm{ref}})`
    clipped to the bounds, and five values of :math:`K_2` spread evenly over
    the allowed interval at :math:`K_1 = K_1^{\\mathrm{ref}}`. The start with the
    smallest residual is kept, the residual being the larger of the two absolute
    modulus differences.

    Parameters
    ----------
    k1_ref : float
        Reference first-harmonic coupling (rad/s).
    k2_ref : float
        Reference second-harmonic coupling (rad/s); must be nonzero.
    k2_bound : float, default 0.15
        Largest :math:`|K_2|` allowed for the partner (rad/s).
    tolerance : float, default 1e-6
        Largest modulus difference for which the pair counts as found.
    k1_search_max : float, default 3.0
        Upper bound on the partner's :math:`K_1` (rad/s).
    dt : float or None, default None
        Step of the simulated scheme to match on (s), or None to match the
        continuous-time model.
    **model : float
        Forwarded to :func:`stationary_moments` or
        :func:`discretised_stationary_moments`: ``omega_1``, ``omega_2``,
        ``noise_std`` and ``n_modes``.

    Returns
    -------
    pair : EqualModuliPair
        The reference, the best partner found, its residual (on the matched
        law) and whether the residual is within ``tolerance`` with a partner
        :math:`K_2` strictly of the opposite sign. A request out of reach is
        returned with ``found=False``, not raised, and a warning is logged.

    Raises
    ------
    ValueError
        If a keyword of ``model`` is not one of the four listed above (the
        message names it), if ``dt`` is given but not finite and positive, if
        ``k2_ref`` is zero (the opposite sign is then undefined), or if
        ``k2_bound``, ``tolerance`` or ``k1_search_max`` is not positive.

    Notes
    -----
    When to use: to build two settings that agree on :math:`|m_1|` and
    :math:`|m_2|` and differ only in the relative phase of the two moments
    (:func:`relative_phase`), so that a statistic that tells them apart must
    read that phase. Some references have no partner within the bound; with
    the default model, :math:`(K_1, K_2) = (0.05, 0.15)` is one, and the
    result then carries ``found=False``. A local search from a few starts can
    miss a partner that exists; ``found=False`` means none was found, not that
    none exists. Match on the scheme (``dt`` set to the simulation step) when
    the pair will be simulated: a pair matched in continuous time is, at the
    bench step 0.05, mismatched by several times ``1e-4`` on the moduli.

    Examples
    --------
    >>> pair = solve_equal_moduli_pair(0.05, 0.12)
    >>> pair.found, pair.k2 < 0
    (True, True)
    """
    _check_model_keywords(model)
    _check_matching_step(dt)
    if k2_ref == 0.0:
        raise ValueError("k2_ref must be nonzero: the opposite sign is undefined")
    if k2_bound <= 0.0:
        raise ValueError(f"k2_bound must be positive, got {k2_bound}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if k1_search_max <= 0.0:
        raise ValueError(f"k1_search_max must be positive, got {k1_search_max}")

    ref = np.abs(
        _matched_moments(k1_ref, k2_ref, max_harmonic=2, dt=dt, model=model)[1:3]
    )

    def differences(x: NDArray[np.float64]) -> NDArray[np.float64]:
        moments = _matched_moments(x[0], x[1], max_harmonic=2, dt=dt, model=model)
        return np.abs(moments[1:3]) - ref

    sign = float(np.sign(k2_ref))
    far = -sign * k2_bound
    lower = np.array([0.0, min(far, 0.0)])
    upper = np.array([k1_search_max, max(far, 0.0)])
    k1_start = float(np.clip(k1_ref, 0.0, k1_search_max))
    # Interior fractions of the K2 interval, away from both K2 = 0 and the bound.
    fractions = (np.arange(_N_PAIR_STARTS) + 0.5) / _N_PAIR_STARTS
    starts = [(k1_start, float(np.clip(-k2_ref, lower[1], upper[1])))]
    starts += [(k1_start, float(far * f)) for f in fractions]

    best_x, residual = None, np.inf
    for start in starts:
        fit = least_squares(
            differences,
            np.array(start),
            bounds=(lower, upper),
            xtol=_PAIR_SOLVER_TOL,
            ftol=_PAIR_SOLVER_TOL,
            gtol=_PAIR_SOLVER_TOL,
        )
        # Select on the quantity `found` is judged on, not on the squared cost.
        fit_residual = float(np.abs(differences(fit.x)).max())
        if best_x is None or fit_residual < residual:
            best_x, residual = fit.x, fit_residual
    k1, k2 = (float(v) for v in best_x)
    # The interval for K2 includes 0, so the opposite sign is checked, not assumed.
    opposite_sign = np.sign(k2) == -sign
    found = bool(residual <= tolerance and opposite_sign)
    if not found:
        logger.warning(
            "no equal-moduli partner of (k1=%s, k2=%s) with |k2| <= %s: closest is "
            "(k1=%.6f, k2=%.6g) with residual %.3e (tolerance %.1e) and k2 of "
            "opposite sign: %s",
            k1_ref,
            k2_ref,
            k2_bound,
            k1,
            k2,
            residual,
            tolerance,
            bool(opposite_sign),
        )
    return EqualModuliPair(
        k1_ref=float(k1_ref),
        k2_ref=float(k2_ref),
        k1=k1,
        k2=k2,
        residual=residual,
        found=found,
    )
