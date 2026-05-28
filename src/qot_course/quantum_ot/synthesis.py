"""S16 synthesis --- the course's greatest hits in one callable place.

Each session of the course delivered a numerical promise. This module re-runs the
canonical results of S5 through S14 in a single function, returning a dictionary keyed
by the session and the result. Useful both for the closing session's demonstration and
for regression testing the whole course.
"""

from __future__ import annotations

import numpy as np
import ot

from qot_course.geometry.info_geometry import (
    fisher_rao_distance,
    mixture_interpolation,
    wasserstein_interpolation_1d,
)
from qot_course.infotheory.classical import (
    kl_divergence,
    mutual_information,
    shannon_entropy,
)
from qot_course.infotheory.quantum import (
    bures_distance,
    quantum_conditional_entropy,
    quantum_mutual_information,
    quantum_relative_entropy,
)
from qot_course.quantum.composite import bell_state
from qot_course.quantum.density import (
    density_matrix,
    maximally_mixed,
)
from qot_course.quantum.states import KET_0, KET_1, KET_PLUS
from qot_course.quantum_ot.sdp import (
    quadratic_position_cost,
    quantum_ot_sdp,
    quantum_wasserstein_squared_swap,
)
from qot_course.quantum_ot.sinkhorn import (
    matrix_gibbs_kernel,
    quantum_sinkhorn_sdp,
    umegaki_kl_to_kernel,
)
from qot_course.transport.discrete import discrete_ot_plan, squared_euclidean_cost
from qot_course.transport.gaussian import bures_matrix_distance
from qot_course.transport.wasserstein import wasserstein_1d


def course_greatest_hits() -> dict[str, float]:
    """Re-run every canonical numerical result of the course (S5--S14).

    Returns a dictionary mapping a human-readable label to the computed value.
    Each entry corresponds to a claim that was proved or verified in one session and
    serves as a regression test for the whole pipeline.
    """
    results: dict[str, float] = {}

    # S5 --- classical information theory.
    results["S5: H(fair coin) [bits]"] = shannon_entropy([0.5, 0.5])
    results["S5: D(p || q) > 0 [bits]"] = kl_divergence([0.7, 0.3], [0.4, 0.6])
    indep = np.outer([0.5, 0.5], [0.5, 0.5])
    correlated = np.array([[0.5, 0.0], [0.0, 0.5]])
    results["S5: I(independent) [bits]"] = mutual_information(indep)
    results["S5: I(correlated) [bits]"] = mutual_information(correlated)

    # S6 --- information geometry.
    results["S6: d_FR(uniform, peaked) [rad]"] = fisher_rao_distance(
        [1 / 3, 1 / 3, 1 / 3], [0.7, 0.2, 0.1]
    )
    support = np.linspace(0.0, 24.0, 200)

    def bump(c: float) -> np.ndarray:
        b = np.exp(-0.5 * ((support - c) / 1.2) ** 2)
        return b / b.sum()

    pt_mix = mixture_interpolation(bump(4.0), bump(20.0), 0.5)
    pt_w2 = wasserstein_interpolation_1d(bump(4.0), bump(20.0), support, 0.5)
    results["S6: mixture midpoint mass in middle"] = float(
        pt_mix[(support >= 8) & (support <= 16)].sum()
    )
    results["S6: W2 midpoint mass in middle"] = float(
        pt_w2[(support >= 8) & (support <= 16)].sum()
    )

    # S7 --- quantum information theory.
    rho_bell = density_matrix(bell_state())
    results["S7: Bell QMI [bits]"] = quantum_mutual_information(
        rho_bell, dims=[2, 2]
    )
    results["S7: Bell S(A|B) [bits]"] = quantum_conditional_entropy(
        rho_bell, dims=[2, 2]
    )
    plus = density_matrix(KET_PLUS)
    mixed = maximally_mixed(2)
    results["S7: S(|+><+| || I/2) [bits]"] = quantum_relative_entropy(plus, mixed)
    results["S7: d_B(|+><+|, I/2)"] = bures_distance(plus, mixed)

    # S8 --- Monge --> Kantorovich.
    a = np.array([0.5, 0.5])
    b = np.array([1 / 3, 1 / 3, 1 / 3])
    cost = squared_euclidean_cost([0.0, 4.0], [1.0, 2.0, 3.0])
    plan = discrete_ot_plan(a, b, cost)
    results["S8: monge-fails LP cost"] = float(np.sum(plan * cost))

    # S9 --- Wasserstein.
    p_vec = np.array([0.7, 0.2, 0.1])
    q_vec = np.array([0.1, 0.3, 0.6])
    positions = np.arange(3, dtype=float)
    w2_cf = wasserstein_1d(positions, p_vec, positions, q_vec, p=2)
    cost_3 = squared_euclidean_cost(positions, positions)
    w2_lp = float(np.sqrt(ot.emd2(p_vec, q_vec, cost_3)))
    results["S9: W2 closed form"] = w2_cf
    results["S9: W2 via LP"] = w2_lp

    # S10 --- classical Sinkhorn (LP limit).
    # We exercise the entropic SDP indirectly via the quantum Sinkhorn module on
    # diagonal states (commuting case = classical limit).
    rho_p = np.diag(p_vec[:2] / p_vec[:2].sum()).astype(complex)
    rho_q = np.diag(q_vec[:2] / q_vec[:2].sum()).astype(complex)
    C2 = quadratic_position_cost([0.0, 1.0])
    sk_value, sk_plan = quantum_sinkhorn_sdp(rho_p, rho_q, C2, epsilon=0.05)
    results["S10/S14: classical Sinkhorn limit (diag, small eps)"] = float(
        np.real(np.trace(C2 @ sk_plan))
    )

    # S11 --- Bures-Wasserstein bridge.
    bw_matrix_term = bures_matrix_distance(plus, mixed)
    results["S11: sqrt(BW matrix) (= d_B from S7)"] = float(
        np.sqrt(max(0.0, bw_matrix_term))
    )

    # S13 --- QOT SDP.
    qot_value, _ = quantum_ot_sdp(plus, mixed, C2)
    results["S13: QOT(|+><+|, I/2) > 0"] = qot_value
    results["S13: SWAP-QOT^2(|0>, |1>) = 1"] = quantum_wasserstein_squared_swap(
        density_matrix(KET_0), density_matrix(KET_1)
    )

    # S14 --- Amari quantum bridge identity (LHS vs RHS).
    eps = 0.4
    _, plan_eps = quantum_sinkhorn_sdp(plus, mixed, C2, epsilon=eps)
    K = matrix_gibbs_kernel(C2, eps)
    transport = float(np.real(np.trace(C2 @ plan_eps)))
    vals = np.linalg.eigvalsh(0.5 * (plan_eps + plan_eps.conj().T))
    vals = vals[vals > 1e-12]
    s_plan = float(-np.sum(vals * np.log(vals)))
    results["S14: tr(C P) - eps S(P)"] = transport - eps * s_plan
    results["S14: eps * S_Umegaki(P || K)"] = eps * umegaki_kl_to_kernel(
        plan_eps, K
    )

    return results
