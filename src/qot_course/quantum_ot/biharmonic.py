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
stationary law of the phase difference :math:`\\Delta = \\theta_2 - \\theta_1` is
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

import numpy as np
from numpy.typing import NDArray

# Number of noise components per step: one per oscillator.
_N_OSCILLATORS = 2
# A record needs at least two samples to define a phase difference trajectory
# and an increment (structural, not a tuning choice).
_MIN_SAMPLES = 2


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
        If ``dt``, ``duration``, ``n_records`` or ``noise_std`` is not
        positive (``noise_std`` may be zero only if non-negative is meant: it
        must be non-negative), if ``burn_in`` is negative, or if ``duration``
        gives fewer than two samples. The message names the offending argument.

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
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
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
