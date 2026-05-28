"""S15 capstone --- coupling measures on a synthetic Kuramoto dyad.

We simulate two coupled Kuramoto oscillators with stochastic noise, build a bipartite
density matrix from their phases by mapping each oscillator to a phase-coherent qubit
state :math:`|\\psi(t)\\rangle = (|0\\rangle + e^{i\\theta(t)}|1\\rangle)/\\sqrt{2}`,
and compare four candidate coupling measures against the known injected coupling
strength :math:`K`:

- **Quantum mutual information** :math:`I(A{:}B) = S(\\rho_{AB} \\| \\rho_A \\otimes \\rho_B)`
  --- the principal *information-theoretic* coupling measure (S7).
- **Bures-coupling** :math:`d_B(\\rho_{AB}, \\rho_A \\otimes \\rho_B)` --- a
  *transport-theoretic* coupling measure (S11 bridge: Bures = Wasserstein on PSD
  matrices).
- **Phase-locking value** :math:`\\mathrm{PLV} = |\\langle e^{i(\\theta_A - \\theta_B)}\\rangle|`
  --- the standard classical baseline in neuroscience.
- **Classical correlation** :math:`\\mathrm{corr}(\\cos\\theta_A, \\cos\\theta_B)` ---
  Euclidean-style baseline.

References: Y. Kuramoto, *Chemical Oscillations, Waves, and Turbulence* (Springer,
1984); J.-P. Lachaux, E. Rodriguez, J. Martinerie, F. Varela, "Measuring phase
synchrony in brain signals", *Hum. Brain Mapp.* **8**, 194 (1999); Trevisan,
arXiv:2202.02091 (2022).
"""

from __future__ import annotations

import numpy as np

from qot_course.infotheory.quantum import (
    bures_distance,
    quantum_mutual_information,
)
from qot_course.quantum.composite import tensor as quantum_tensor
from qot_course.quantum.composite import partial_trace


def simulate_kuramoto_dyad(
    K: float,
    omega_1: float = 1.0,
    omega_2: float = 1.2,
    duration: float = 200.0,
    dt: float = 0.05,
    noise_std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler--Maruyama simulation of two coupled Kuramoto oscillators with noise.

    SDE: :math:`\\mathrm{d}\\theta_i = (\\omega_i + K \\sin(\\theta_j - \\theta_i))\\,
    \\mathrm{d}t + \\sigma\\,\\mathrm{d}W_i`. Returns ``(theta_1, theta_2)`` of length
    ``int(duration / dt)``.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    theta_1 = np.zeros(n_steps)
    theta_2 = np.zeros(n_steps)
    sqrt_dt = float(np.sqrt(dt))
    for k in range(n_steps - 1):
        d1 = theta_2[k] - theta_1[k]
        d2 = theta_1[k] - theta_2[k]
        theta_1[k + 1] = (
            theta_1[k]
            + dt * (omega_1 + K * np.sin(d1))
            + noise_std * sqrt_dt * rng.standard_normal()
        )
        theta_2[k + 1] = (
            theta_2[k]
            + dt * (omega_2 + K * np.sin(d2))
            + noise_std * sqrt_dt * rng.standard_normal()
        )
    return theta_1, theta_2


def phase_qubit_state(theta: np.ndarray) -> np.ndarray:
    """Map a phase :math:`\\theta` to the qubit state
    :math:`(|0\\rangle + e^{i\\theta}|1\\rangle)/\\sqrt{2}`.

    Returns a shape ``(len(theta), 2)`` array of complex amplitudes.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    return np.column_stack(
        [np.ones_like(theta, dtype=complex), np.exp(1j * theta)]
    ) / np.sqrt(2.0)


def joint_density_matrix(
    theta_1: np.ndarray, theta_2: np.ndarray
) -> np.ndarray:
    """Build the time-averaged bipartite density matrix
    :math:`\\rho_{AB} = \\mathbb{E}_t[|\\psi_A(t)\\rangle\\langle\\psi_A(t)|
    \\otimes |\\psi_B(t)\\rangle\\langle\\psi_B(t)|]`.

    For uncoupled oscillators with uniformly drifting phases this concentrates onto
    :math:`I_4 / 4`. For phase-locked oscillators the off-block coherence
    :math:`(01, 10)` becomes nonzero --- the same quantity that drives PLV.
    """
    psi_a = phase_qubit_state(theta_1)
    psi_b = phase_qubit_state(theta_2)
    # |Psi(t)> = psi_a(t) ⊗ psi_b(t)
    psi_joint = np.einsum("ti,tj->tij", psi_a, psi_b).reshape(-1, 4)
    rho_ab = (psi_joint.T @ psi_joint.conj()) / psi_joint.shape[0]
    # Hermitize to absorb floating-point asymmetry from einsum.
    return 0.5 * (rho_ab + rho_ab.conj().T)


def plv(theta_1: np.ndarray, theta_2: np.ndarray) -> float:
    """Phase-locking value :math:`|\\langle e^{i(\\theta_1 - \\theta_2)}\\rangle|`.

    Bounded in [0, 1]: 0 = phases independent, 1 = perfect phase locking.
    """
    return float(
        np.abs(np.mean(np.exp(1j * (np.asarray(theta_1) - np.asarray(theta_2)))))
    )


def cosine_correlation(theta_1: np.ndarray, theta_2: np.ndarray) -> float:
    """Pearson correlation between :math:`\\cos\\theta_1` and :math:`\\cos\\theta_2`.

    Classical (Euclidean) coupling baseline.
    """
    c1 = np.cos(np.asarray(theta_1))
    c2 = np.cos(np.asarray(theta_2))
    return float(np.corrcoef(c1, c2)[0, 1])


def coupling_qmi(rho_ab: np.ndarray, dims: tuple[int, int] = (2, 2)) -> float:
    """Quantum mutual information :math:`I(A{:}B) = S(\\rho_{AB} \\| \\rho_A \\otimes \\rho_B)`.

    Uses nats (natural log) for consistency with cvxpy. Convert to bits by dividing
    by :math:`\\log 2`.
    """
    return quantum_mutual_information(rho_ab, dims=list(dims), base=np.e)


def coupling_bures(
    rho_ab: np.ndarray, dims: tuple[int, int] = (2, 2)
) -> float:
    """Bures-coupling :math:`d_B(\\rho_{AB}, \\rho_A \\otimes \\rho_B)`.

    A transport-theoretic coupling measure (S11 bridge). Zero iff
    :math:`\\rho_{AB} = \\rho_A \\otimes \\rho_B`.
    """
    rho_a = partial_trace(rho_ab, keep=[0], dims=list(dims))
    rho_b = partial_trace(rho_ab, keep=[1], dims=list(dims))
    product = quantum_tensor(rho_a, rho_b)
    return bures_distance(rho_ab, product)
