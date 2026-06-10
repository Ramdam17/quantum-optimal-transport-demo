"""Quantum optimal transport as a semidefinite program (SDP).

S13 promotes the Kantorovich LP of S8 to a genuine operator-level problem. A
**quantum coupling** of two states :math:`\\rho_A \\in \\mathcal{S}(\\mathcal{H}_A),
\\rho_B \\in \\mathcal{S}(\\mathcal{H}_B)` is a bipartite density matrix
:math:`\\rho_{AB}` on :math:`\\mathcal{H}_A \\otimes \\mathcal{H}_B` whose partial
traces are :math:`\\rho_A` and :math:`\\rho_B`. Given a Hermitian cost operator
:math:`C` the quantum OT problem is

.. math::
    \\mathrm{QOT}(\\rho_A, \\rho_B) = \\min_{\\rho_{AB} \\in \\Pi(\\rho_A, \\rho_B)}
        \\mathrm{tr}(C\\,\\rho_{AB}).

This is a **semidefinite program**: a linear objective on the bipartite Hermitian
variable, with linear partial-trace constraints and the PSD cone constraint
:math:`\\rho_{AB} \\succeq 0`. cvxpy supports it directly via :func:`cvxpy.partial_trace`.

Two canonical cost operators appear in the literature:

- **SWAP cost** :math:`C = I - \\mathrm{SWAP}` (Cole, Lostaglio, Verma, Wilde, 2023).
  For pure states the SDP returns :math:`1 - |\\langle\\psi|\\phi\\rangle|^2`.
- **Quadratic position cost** :math:`C = (X_A \\otimes I - I \\otimes X_B)^2` for some
  diagonal "position" observable :math:`X`. For diagonal :math:`\\rho_A, \\rho_B` in the
  :math:`X` eigenbasis the SDP returns the classical :math:`W_2^2` with cost
  :math:`(i - j)^2` --- the diagonal-collapse principle of S12.

References: G. De Palma, D. Trevisan, "Quantum optimal transport with quantum
channels", Ann. Henri Poincare 22, 3199 (2021); S. Cole, M. Lostaglio, K. Verma,
M. M. Wilde, "Quantum Wasserstein distance based on an optimization over separable
states", IEEE Trans. Inf. Theory (2023); D. Trevisan, "Optimal transport methods for
quantum systems", arXiv:2202.02091 (2022).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def swap_matrix(d: int) -> np.ndarray:
    """SWAP operator on :math:`\\mathcal{H} \\otimes \\mathcal{H}` of dim :math:`d`.

    Acts on the computational basis by :math:`\\mathrm{SWAP}\\,|i\\rangle|j\\rangle =
    |j\\rangle|i\\rangle`. Hermitian and unitary (involution).
    """
    swap = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            # |i,j> sits at row index i*d + j (numpy's Kronecker convention).
            swap[j * d + i, i * d + j] = 1.0
    return swap


def swap_cost(d: int) -> np.ndarray:
    """Cost operator :math:`C = I - \\mathrm{SWAP}` on :math:`\\mathcal{H}^{\\otimes 2}`.

    Used by Cole et al. (2023). For pure-state inputs the QOT SDP returns
    :math:`1 - |\\langle\\psi|\\phi\\rangle|^2`.
    """
    return np.eye(d * d, dtype=complex) - swap_matrix(d)


def quadratic_position_cost(positions: np.ndarray) -> np.ndarray:
    """Cost :math:`C = (X_A \\otimes I - I \\otimes X_B)^2` for the diagonal
    "position" observable :math:`X = \\mathrm{diag}(\\text{positions})`.

    Reduces to the classical :math:`W_2^2` cost matrix :math:`C_{ij} = (i - j)^2` on
    diagonal density matrices in the :math:`X` eigenbasis.
    """
    positions = np.asarray(positions, dtype=float)
    d = positions.shape[0]
    X = np.diag(positions)
    I_d = np.eye(d, dtype=float)
    diff = np.kron(X, I_d) - np.kron(I_d, X)
    return (diff @ diff).astype(complex)


def quantum_ot_sdp(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    cost: np.ndarray,
    solver: str | None = None,
) -> tuple[float, np.ndarray]:
    """Solve the quantum optimal-transport SDP and return ``(value, plan)``.

    Minimises :math:`\\mathrm{tr}(C\\,\\rho_{AB})` over Hermitian PSD couplings
    :math:`\\rho_{AB}` with marginals :math:`\\mathrm{tr}_B(\\rho_{AB}) = \\rho_A` and
    :math:`\\mathrm{tr}_A(\\rho_{AB}) = \\rho_B`. The returned plan is a complex
    :math:`d_A d_B \\times d_A d_B` Hermitian PSD matrix.

    Parameters
    ----------
    rho_a, rho_b : array_like
        Density matrices for the two marginals.
    cost : array_like
        Hermitian cost operator on :math:`\\mathcal{H}_A \\otimes \\mathcal{H}_B`.
    solver : str, optional
        cvxpy solver name (e.g. ``"SCS"``, ``"CLARABEL"``). Defaults to cvxpy's choice.
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    cost = np.asarray(cost, dtype=complex)
    d_a = rho_a.shape[0]
    d_b = rho_b.shape[0]

    plan = cp.Variable((d_a * d_b, d_a * d_b), hermitian=True)
    constraints = [
        plan >> 0,
        cp.partial_trace(plan, (d_a, d_b), axis=1) == rho_a,
        cp.partial_trace(plan, (d_a, d_b), axis=0) == rho_b,
    ]
    objective = cp.Minimize(cp.real(cp.trace(cost @ plan)))
    problem = cp.Problem(objective, constraints)
    # CLARABEL is the interior-point SDP solver shipped with cvxpy. We request very
    # tight tolerances (1e-10) because the default settings hover around 1e-4, far too
    # loose for closed-form validation tests. CLARABEL frequently cannot *certify* a gap
    # this small and returns status "optimal_inaccurate" (with a UserWarning we leave
    # visible, not silenced); in practice the returned value is accurate to ~1e-7--1e-8,
    # which all the closed-form regression tests confirm. We accept that status below.
    chosen_solver = solver if solver is not None else "CLARABEL"
    if chosen_solver == "CLARABEL":
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10)
    else:
        problem.solve(solver=chosen_solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QOT SDP did not converge (status: {problem.status}).")
    return float(problem.value), np.asarray(plan.value, dtype=complex)


def quantum_wasserstein_squared_swap(
    rho_a: np.ndarray, rho_b: np.ndarray, solver: str | None = None
) -> float:
    """Quantum-Wasserstein-:math:`^2` with SWAP cost (Cole et al., 2023).

    Equivalent to :math:`\\mathrm{QOT}(\\rho_A, \\rho_B)` with :math:`C = I - \\mathrm{SWAP}`.
    """
    if rho_a.shape != rho_b.shape:
        raise ValueError("SWAP cost requires equal-dimension subsystems.")
    d = rho_a.shape[0]
    value, _ = quantum_ot_sdp(rho_a, rho_b, swap_cost(d), solver=solver)
    return value
