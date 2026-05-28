"""Quantum entropic optimal transport --- the operator-level Sinkhorn.

S14 lifts S10's entropic regularisation to density matrices:

.. math::
    \\rho^\\star_{AB,\\,\\varepsilon} = \\arg\\min_{\\rho_{AB} \\in \\Pi(\\rho_A, \\rho_B)}
    \\ \\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\, S(\\rho_{AB}),

with :math:`S(\\rho_{AB}) = -\\mathrm{tr}(\\rho_{AB}\\log\\rho_{AB})` (von Neumann entropy).
The **matrix-exponential Gibbs kernel** :math:`K = \\exp(-C/\\varepsilon)` plays the role
of the entrywise classical kernel.

\\textbf{Quantum Amari bridge}. A one-line algebra reveals

.. math::
    \\varepsilon\\,S_{\\mathrm{Umegaki}}(\\rho_{AB}\\,\\|\\,K)
    = \\mathrm{tr}(\\rho_{AB}\\,\\log\\rho_{AB}) - \\mathrm{tr}(\\rho_{AB}\\,\\log K)\\cdot\\varepsilon / 1
    = \\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\,S(\\rho_{AB}),

so the entropic QOT plan is the **Umegaki relative-entropy projection** of the
matrix-exponential Gibbs kernel onto the coupling polytope. The Sinkhorn iterations of
S10 are *iterative Bregman projections* under the KL divergence; their quantum analogue
is the operator Bregman iteration under the Umegaki divergence (Peyre, Chizat, Vialard,
Schmitzer, 2019; Pelikh, Gerolin et al.).

For pedagogical clarity we solve the entropic SDP directly via cvxpy's
:func:`cvxpy.von_neumann_entr` atom rather than running the explicit operator iteration.
Both approaches yield the same plan.

References: M. Cuturi, "Sinkhorn distances", NeurIPS (2013); G. Peyre, L. Chizat, F.-X.
Vialard, B. Schmitzer, "Quantum entropic regularization of matrix-valued optimal
transport", European J. Appl. Math. 30, 1079 (2019); D. Trevisan, arXiv:2202.02091
(2022); cvxpy docs.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import scipy.linalg as spla


def matrix_gibbs_kernel(cost: np.ndarray, epsilon: float) -> np.ndarray:
    """Matrix-exponential Gibbs kernel :math:`K = \\exp(-C/\\varepsilon)`.

    Quantum analogue of the entrywise classical Gibbs kernel
    :math:`K_{ij} = e^{-C_{ij}/\\varepsilon}` from S10. For commuting (diagonal) cost
    operators the two coincide; in general the matrix exponential adds operator-mixing.
    """
    return spla.expm(-np.asarray(cost, dtype=complex) / float(epsilon))


def quantum_sinkhorn_sdp(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    solver: str | None = None,
) -> tuple[float, np.ndarray]:
    """Solve the von-Neumann-entropy-regularised quantum OT SDP.

    Minimises :math:`\\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\,S(\\rho_{AB})` over
    Hermitian PSD couplings with prescribed partial-trace marginals. Returns the
    optimal *objective value* (transport cost minus :math:`\\varepsilon\\,S`) and the
    optimal plan.

    Note
    ----
    cvxpy's :func:`cvxpy.von_neumann_entr` uses the *natural log* convention; the
    classical Sinkhorn module's :math:`\\varepsilon` in S10 was in the same convention,
    so the comparison is direct.
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    cost = np.asarray(cost, dtype=complex)
    d_a, d_b = rho_a.shape[0], rho_b.shape[0]

    plan = cp.Variable((d_a * d_b, d_a * d_b), hermitian=True)
    constraints = [
        plan >> 0,
        cp.partial_trace(plan, (d_a, d_b), axis=1) == rho_a,
        cp.partial_trace(plan, (d_a, d_b), axis=0) == rho_b,
    ]
    transport = cp.real(cp.trace(cost @ plan))
    entropy = cp.von_neumann_entr(plan)
    objective = cp.Minimize(transport - epsilon * entropy)
    problem = cp.Problem(objective, constraints)

    chosen_solver = solver if solver is not None else "CLARABEL"
    if chosen_solver == "CLARABEL":
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9)
    else:
        problem.solve(solver=chosen_solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Quantum Sinkhorn SDP did not converge ({problem.status})."
        )
    return float(problem.value), np.asarray(plan.value, dtype=complex)


def quantum_sinkhorn_cost(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    solver: str | None = None,
) -> float:
    """Transport cost ``tr(C * P_eps)`` (without the entropy term) at the entropic
    optimum --- useful for comparing to the non-regularised SDP of S13."""
    _, plan = quantum_sinkhorn_sdp(rho_a, rho_b, cost, epsilon, solver=solver)
    return float(np.real(np.trace(np.asarray(cost) @ plan)))


def umegaki_kl_to_kernel(
    plan: np.ndarray, kernel: np.ndarray, atol: float = 1e-12
) -> float:
    """Umegaki-type relative entropy :math:`S(P \\| K) = \\mathrm{tr}(P (\\log P - \\log K))`.

    For Hermitian PSD ``plan`` and Hermitian PSD ``kernel`` (the matrix exponential
    Gibbs kernel of :func:`matrix_gibbs_kernel`). The classical KL identity
    :math:`\\varepsilon\\,\\mathrm{KL}(P \\| K) = \\langle C, P\\rangle - \\varepsilon H(P)`
    lifts to :math:`\\varepsilon\\,S_{\\mathrm{Umegaki}}(P \\| K) = \\mathrm{tr}(C\\,P)
    - \\varepsilon\\,S(P)` --- this is the **Amari quantum bridge** of S14.
    """
    plan = 0.5 * (
        np.asarray(plan, dtype=complex) + np.asarray(plan, dtype=complex).conj().T
    )
    kernel = 0.5 * (
        np.asarray(kernel, dtype=complex)
        + np.asarray(kernel, dtype=complex).conj().T
    )

    def _matrix_log(matrix: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(matrix)
        log_vals = np.where(
            vals > atol, np.log(np.where(vals > atol, vals, 1.0)), 0.0
        )
        return (vecs * log_vals) @ vecs.conj().T

    return float(
        np.real(np.trace(plan @ (_matrix_log(plan) - _matrix_log(kernel))))
    )
