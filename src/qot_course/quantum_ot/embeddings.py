"""The embedding menu --- mapping classical phase/amplitude signals to quantum states.

Each embedding preserves different structure of the signal. The *naive phase qubit*
(:func:`qot_course.quantum_ot.capstone.phase_qubit_state`) keeps only the first circular
moment of the phase --- exactly what PLV sees; the richer embeddings here keep more.

The four embeddings in this module:

- :func:`amplitude_phase_state` --- a qubit whose *polar angle* carries the instantaneous
  amplitude envelope and whose *azimuth* carries the phase. Keeps amplitude information
  that the pure-phase qubit discards.
- :func:`covariance_density` --- the (normalised) :math:`2\\times 2` second-moment matrix of
  the analytic-phase unit vector :math:`(\\cos\\theta, \\sin\\theta)`. A *mixed* state
  (an ensemble average), not a pure state: it forgets per-sample phase but keeps the
  circular concentration / mean-resultant direction.
- :func:`multifreq_state` --- a qudit stacking phase *harmonics*
  :math:`(1, e^{i\\theta}, e^{2i\\theta}, \\dots)`. With the default harmonics ``(1, 2)`` this is a
  qutrit carrying the **first AND second** circular moments of the phase, so the joint
  density (below) exposes both :math:`\\langle e^{i\\delta}\\rangle` (what PLV sees) and
  :math:`\\langle e^{2i\\delta}\\rangle` (which PLV is blind to) --- the structural reason a
  QOT coupling measure can separate signals that PLV cannot.
- :func:`joint_density_from_states` --- the general time-averaged bipartite density matrix
  :math:`\\mathbb{E}_t[|\\psi_A\\psi_B\\rangle\\langle\\psi_A\\psi_B|]` for *any* per-sample
  embeddings (the qudit generalisation of
  :func:`qot_course.quantum_ot.capstone.joint_density_matrix`).

Index convention (shared with :mod:`qot_course.quantum.composite`): the basis vector
:math:`|i\\rangle_A |j\\rangle_B` sits at flat index ``i * d_B + j`` (Kronecker / row-major
order), so the joint amplitude ``psi[:, i * d_B + j] = psi_a[:, i] * psi_b[:, j]``.

References
----------
J.-P. Lachaux, E. Rodriguez, J. Martinerie, F. J. Varela, "Measuring phase synchrony in
brain signals", *Hum. Brain Mapp.* **8**, 194 (1999),
DOI:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C.
K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
DOI:10.1002/9780470316979.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def amplitude_phase_state(theta: ArrayLike, amp: ArrayLike) -> NDArray[np.complex128]:
    r"""Embed (phase, amplitude) samples as qubit states keeping both quantities.

    Each sample maps to :math:`\cos\alpha\,|0\rangle + \sin\alpha\,e^{i\theta}\,|1\rangle`,
    where the polar angle :math:`\alpha = \tfrac{\pi}{2}\,\mathrm{clip}(a, 0, 1)` encodes the
    (clipped, unit-scaled) amplitude envelope :math:`a` and the azimuth encodes the phase
    :math:`\theta`. Amplitude ``0`` gives :math:`|0\rangle` (no phase weight); amplitude ``1``
    gives :math:`e^{i\theta}|1\rangle` (the full phase qubit of
    :func:`qot_course.quantum_ot.capstone.phase_qubit_state`, up to global phase).

    Parameters
    ----------
    theta : array_like, shape (N,)
        Instantaneous phase per sample, in radians. Flattened on input.
    amp : array_like, shape (N,)
        Instantaneous amplitude envelope per sample, in **normalised units** (expected in
        ``[0, 1]``; values outside are clipped). Flattened on input.

    Returns
    -------
    psi : ndarray of complex, shape (N, 2)
        Unit-norm qubit amplitudes; ``psi[t]`` is the state for sample ``t``. Each row
        satisfies :math:`\sum_k |\psi_{tk}|^2 = 1` exactly (since
        :math:`\cos^2\alpha + \sin^2\alpha = 1`).

    Notes
    -----
    When to use
        Use when the amplitude envelope carries coupling information you do not want to
        throw away --- e.g. amplitude-amplitude or phase-amplitude coupling. The naive
        phase qubit is *amplitude-blind* (it fixes both basis weights at
        :math:`1/\sqrt{2}`); this embedding lets the :math:`|1\rangle` weight track power,
        so a flat-power and a bursty signal with identical phases yield *different* states.
        For pure phase synchrony, prefer the lighter phase qubit or
        :func:`multifreq_state`.

    The amplitude-to-angle map is the standard Bloch-sphere parametrisation
    :math:`|\psi\rangle = \cos\tfrac{\vartheta}{2}|0\rangle + e^{i\varphi}\sin\tfrac{\vartheta}{2}|1\rangle`
    with polar angle :math:`\vartheta = 2\alpha`; clipping to :math:`[0, 1]` keeps
    :math:`\alpha \in [0, \pi/2]` so amplitude maps monotonically onto one Bloch meridian.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> psi = amplitude_phase_state(theta, amp=np.abs(np.cos(theta)))
    >>> np.allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0)
    True

    References
    ----------
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    # Map amplitude in [0, 1] onto the polar angle alpha in [0, pi/2]: amp 0 -> |0>, amp 1 -> |1>.
    alpha = 0.5 * np.pi * np.clip(np.asarray(amp, dtype=float).ravel(), 0.0, 1.0)
    return np.column_stack(
        [np.cos(alpha).astype(complex), np.sin(alpha) * np.exp(1j * theta)]
    )


def covariance_density(theta: ArrayLike) -> NDArray[np.complex128]:
    r"""Build the normalised second-moment (covariance) density of the phase unit vector.

    Maps each phase sample to the real unit vector :math:`v(\theta) = (\cos\theta, \sin\theta)`
    and returns the trace-normalised ensemble second moment
    :math:`\rho = \mathbb{E}_t[v\,v^{\mathsf T}] / \operatorname{tr}\mathbb{E}_t[v\,v^{\mathsf T}]`.
    This is a :math:`2\times 2` real, symmetric, positive-semidefinite, unit-trace matrix
    --- a valid (generally **mixed**) qubit density matrix.

    Parameters
    ----------
    theta : array_like, shape (N,)
        Phase samples in radians. Flattened on input.

    Returns
    -------
    rho : ndarray of complex, shape (2, 2)
        Hermitian (here real), unit-trace, PSD density matrix. Its anisotropy (eigenvalue
        split) reflects how concentrated the phases are: a single fixed phase gives a rank-1
        (pure) :math:`\rho`; phases spread uniformly over the circle give
        :math:`\tfrac{1}{2} I` (maximally mixed).

    Notes
    -----
    When to use
        Use when you want a *state-of-the-ensemble* rather than a per-sample state --- i.e.
        you care about the circular concentration and mean direction of the phase
        distribution, not the individual samples. Unlike :func:`amplitude_phase_state` and
        :func:`multifreq_state` (which return one *pure* state per sample), this returns a
        single *mixed* :math:`2\times 2` density for the whole record. Use it when a
        compact, channel-/noise-aware summary of phase dispersion is wanted; use the
        per-sample embeddings when you need the joint state of two signals.

    The eigenvector of the leading eigenvalue points along :math:`(\cos\bar\theta,
    \sin\bar\theta)` for the doubled mean angle (the axial mean of directional statistics);
    the eigenvalue gap relates to the mean resultant length :math:`R` via
    :math:`\lambda_\pm = \tfrac{1}{2}(1 \pm \rho_2)` with :math:`\rho_2` the second
    trigonometric moment. See Mardia & Jupp, ch. 3.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> rho = covariance_density(theta)
    >>> rho.shape
    (2, 2)
    >>> bool(abs(np.trace(rho).real - 1.0) < 1e-12)
    True

    References
    ----------
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    v = np.column_stack([np.cos(theta), np.sin(theta)])
    cov = (v.T @ v) / v.shape[0]
    return (cov / np.trace(cov)).astype(complex)


def multifreq_state(
    theta: ArrayLike, harmonics: Sequence[int] = (1, 2)
) -> NDArray[np.complex128]:
    r"""Embed phase samples as a qudit stacking phase harmonics (the multi-frequency state).

    Each sample maps to the equal-weight superposition
    :math:`\frac{1}{\sqrt{d}}\bigl(|0\rangle + \sum_{h} e^{i h \theta}\,|h\rangle\bigr)`,
    where the sum runs over ``harmonics`` and :math:`d = 1 + \mathrm{len(harmonics)}`. The
    constant :math:`|0\rangle` term is the DC reference; basis vector :math:`|k\rangle`
    (for the :math:`k`-th harmonic :math:`h`) carries :math:`e^{i h \theta}`.

    Parameters
    ----------
    theta : array_like, shape (N,)
        Phase samples in radians. Flattened on input.
    harmonics : sequence of int, optional
        Phase harmonics to encode (default ``(1, 2)``: first and second circular moments).
        The resulting qudit dimension is ``d = 1 + len(harmonics)`` --- e.g. ``(1, 2)``
        gives a **qutrit** (``d = 3``); ``(1, 2, 3)`` gives ``d = 4``; an empty sequence
        gives a trivial 1-dimensional constant state.

    Returns
    -------
    psi : ndarray of complex, shape (N, d)
        Unit-norm qudit amplitudes per sample, with ``d = 1 + len(harmonics)``. Column ``0``
        is the constant :math:`1/\sqrt{d}`; column ``k >= 1`` is
        :math:`e^{i\,\mathrm{harmonics}[k-1]\,\theta}/\sqrt{d}`. Each row satisfies
        :math:`\sum_k |\psi_{tk}|^2 = 1` exactly (equal-weight superposition of ``d`` unit
        phasors).

    Notes
    -----
    When to use
        This is the embedding the capstone hinges on. The naive phase qubit and PLV both see
        **only** the first circular moment :math:`\langle e^{i\delta}\rangle` of the phase
        difference :math:`\delta = \theta_A - \theta_B`. Encoding harmonic ``2`` additionally
        exposes the **second** circular moment :math:`\langle e^{2i\delta}\rangle`, which is
        sensitive to phase relationships (e.g. bimodal / anti-phase-symmetric coupling) that
        leave the first moment --- and hence PLV --- unchanged. Feed two such states to
        :func:`joint_density_from_states`: the first moment then lives in the
        :math:`(|1\rangle_A|0\rangle_B,\,|0\rangle_A|1\rangle_B)` coherence and the second in
        :math:`(|2\rangle_A|0\rangle_B,\,|0\rangle_A|2\rangle_B)` --- *distinct* matrix
        elements, so a state-space coupling measure (QMI, Bures) can read structure PLV is
        blind to. Use the lighter phase qubit only when first-moment phase locking is all you
        need; use :func:`amplitude_phase_state` when amplitude (not higher phase harmonics)
        is the missing structure.

    Equal weights keep the embedding faithful to the *directional* content (each harmonic
    contributes equally, mirroring the unweighted circular moments of directional
    statistics); a deliberately unequal weighting would bias the coupling measure toward
    whichever moment is up-weighted.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    >>> psi = multifreq_state(theta, harmonics=(1, 2))
    >>> psi.shape
    (50, 3)
    >>> np.allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0)
    True

    References
    ----------
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    J.-P. Lachaux et al., "Measuring phase synchrony in brain signals",
    *Hum. Brain Mapp.* **8**, 194 (1999),
    DOI:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    # Column 0 is the constant (DC) reference |0>; column k>=1 carries the k-th harmonic.
    cols = [np.ones_like(theta, dtype=complex)]
    cols += [np.exp(1j * h * theta) for h in harmonics]
    # Equal-weight superposition -> divide by sqrt(d) so each per-sample row is unit norm.
    return np.column_stack(cols) / np.sqrt(float(len(cols)))


def joint_density_from_states(
    psi_a: NDArray[np.complex128], psi_b: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    r"""Build the time-averaged bipartite density matrix from per-sample qudit embeddings.

    Forms the product state :math:`|\Psi(t)\rangle = |\psi_A(t)\rangle \otimes
    |\psi_B(t)\rangle` for each sample and returns the empirical average
    :math:`\rho_{AB} = \mathbb{E}_t\bigl[|\Psi(t)\rangle\langle\Psi(t)|\bigr]`. This is the
    general qudit counterpart of
    :func:`qot_course.quantum_ot.capstone.joint_density_matrix` (which is the special case
    ``d_A = d_B = 2``), accepting embeddings of *any* dimension --- qubits, the qutrit
    :func:`multifreq_state`, etc.

    Parameters
    ----------
    psi_a : ndarray of complex, shape (N, d_A)
        Per-sample (generally pure) amplitudes for system :math:`A`; row ``t`` is the state
        at sample ``t``. Rows are assumed unit-norm (as produced by the embeddings in this
        module); they are **not** re-normalised here.
    psi_b : ndarray of complex, shape (N, d_B)
        Per-sample amplitudes for system :math:`B`, sharing the same ``N`` as ``psi_a``.

    Returns
    -------
    rho : ndarray of complex, shape (d_A * d_B, d_A * d_B)
        Hermitian, unit-trace, PSD bipartite density matrix in the
        :math:`|i\rangle_A|j\rangle_B \mapsto i\,d_B + j` (Kronecker / row-major) basis,
        matching :mod:`qot_course.quantum.composite`. Explicitly Hermitised on output to
        absorb the floating-point asymmetry introduced by the outer-product accumulation.

    Notes
    -----
    When to use
        Use to turn two per-sample embeddings (e.g. two :func:`multifreq_state` records) into
        the single joint state on which all the QOT coupling measures (quantum mutual
        information, Bures coupling, entropic QOT) operate. With multi-frequency embeddings,
        the off-diagonal *coherences* expose the cross-signal circular moments: for a
        constant phase difference :math:`\delta = \theta_A - \theta_B`, element
        ``rho[i*d_B + 0, 0*d_B + i]`` carries :math:`\tfrac{1}{d^2}\langle e^{i\,h_i\delta}\rangle`,
        the :math:`h_i`-th circular moment. With ``harmonics=(1, 2)`` (``d = 3``): the first
        moment sits at ``rho[3, 1]`` and the second at ``rho[6, 2]`` (see the module
        docstring's index convention).

    Trace preservation holds because each row of each input is unit-norm, so every
    per-sample :math:`|\Psi(t)\rangle\langle\Psi(t)|` has trace 1 and the average does too.
    PSD holds because the result is an average of (PSD) rank-1 projectors.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ta = rng.uniform(0, 2 * np.pi, 4000)
    >>> tb = rng.uniform(0, 2 * np.pi, 4000)
    >>> rho = joint_density_from_states(multifreq_state(ta), multifreq_state(tb))
    >>> rho.shape
    (9, 9)
    >>> bool(abs(np.trace(rho).real - 1.0) < 1e-9)
    True

    References
    ----------
    Trevisan et al., "An optimal-transport approach to ...", arXiv:2202.02091 (2022).
    K. V. Mardia & P. E. Jupp, *Directional Statistics* (Wiley, 2000),
    DOI:10.1002/9780470316979.
    """
    d_a, d_b = psi_a.shape[1], psi_b.shape[1]
    # |Psi(t)> = psi_a(t) ⊗ psi_b(t); flat index i*d_b + j matches the Kronecker basis order.
    psi = np.einsum("ti,tj->tij", psi_a, psi_b).reshape(-1, d_a * d_b)
    rho = (psi.T @ psi.conj()) / psi.shape[0]
    # Hermitise to absorb floating-point asymmetry from the outer-product accumulation.
    return 0.5 * (rho + rho.conj().T)
