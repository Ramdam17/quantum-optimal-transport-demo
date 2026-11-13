"""Quantum optimal transport as a semidefinite program (SDP).

Notebook ``04/04`` promotes the Kantorovich LP of ``03/03`` to a genuine operator-level
problem. A
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

- **SWAP cost** :math:`C = I - \\mathrm{SWAP}`, twice the cost
  :math:`\\tfrac12 (I - \\mathrm{SWAP})` of Cole, Eckstein, Friedland and Zyczkowski (2023).
  For pure states the SDP returns :math:`1 - |\\langle\\psi|\\phi\\rangle|^2`.
- **Quadratic position cost** :math:`C = (X_A \\otimes I - I \\otimes X_B)^2` for some
  diagonal "position" observable :math:`X`. For diagonal :math:`\\rho_A, \\rho_B` in the
  :math:`X` eigenbasis the SDP returns the classical :math:`W_2^2` with cost
  :math:`(i - j)^2` --- the diagonal-collapse principle of ``04/01``.

References: S. Cole, M. Eckstein, S. Friedland, K. Zyczkowski, "On quantum optimal
transport", Math. Phys. Anal. Geom. 26, 14 (2023), doi:10.1007/s11040-023-09456-7 (the
problem solved here: all quantum couplings, as an SDP); F. Golse, C. Mouhot, T. Paul,
"On the mean field and classical limits of quantum mechanics", Commun. Math. Phys.
343, 165 (2016), doi:10.1007/s00220-015-2485-7 (first transport problem over quantum
couplings); G. De Palma, D. Trevisan, "Quantum optimal transport with quantum
channels", Ann. Henri Poincare 22, 3199 (2021), doi:10.1007/s00023-021-01042-3 (the
neighbouring channel formulation); D. Trevisan, "Quantum optimal transport: an
invitation", Boll. Unione Mat. Ital. 18, 347 (2025), doi:10.1007/s40574-024-00428-5.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from qot_course.quantum_ot.sinkhorn import _pure_marginal_fixes_the_coupling


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

    Twice the cost :math:`\\tfrac12 (I - \\mathrm{SWAP})` of Cole, Eckstein, Friedland and
    Zyczkowski (2023). For pure-state inputs the QOT SDP returns
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
    closed_form: bool = True,
) -> tuple[float, np.ndarray]:
    """Solve the quantum optimal-transport SDP and return ``(value, plan)``.

    Minimises :math:`\\mathrm{tr}(C\\,\\rho_{AB})` over Hermitian PSD couplings
    :math:`\\rho_{AB}` with marginals :math:`\\mathrm{tr}_B(\\rho_{AB}) = \\rho_A` and
    :math:`\\mathrm{tr}_A(\\rho_{AB}) = \\rho_B`.

    When to use: the unregularised transport cost between two density matrices, and the
    reference that the entropic solvers of :mod:`qot_course.quantum_ot.sinkhorn` approach
    as their regularisation vanishes.

    Parameters
    ----------
    rho_a : array_like, shape (d_a, d_a)
        Density matrix of the first marginal (dimensionless, unit trace).
    rho_b : array_like, shape (d_b, d_b)
        Density matrix of the second marginal (dimensionless, unit trace).
    cost : array_like, shape (d_a * d_b, d_a * d_b)
        Hermitian cost operator on :math:`\\mathcal{H}_A \\otimes \\mathcal{H}_B`, in the
        units the transport cost is read in.
    solver : str, optional
        cvxpy solver name (e.g. ``"SCS"``, ``"CLARABEL"``). Defaults to CLARABEL with a
        duality-gap tolerance of 1e-10. Not consulted when the closed form applies.
    closed_form : bool, default True
        When ``True`` and at least one marginal is a pure state, return the product
        coupling and its cost directly, without calling the solver (see the Notes).
        Pass ``False`` to send every input to the numerical solver, which is what a
        check of the solver against the pure-state formula needs.

    Returns
    -------
    value : float
        The minimal transport cost :math:`\\mathrm{tr}(C\\,\\rho_{AB}^\\star)`, in the units
        of ``cost``.
    plan : numpy.ndarray, complex, shape (d_a * d_b, d_a * d_b)
        The optimal coupling :math:`\\rho_{AB}^\\star`, Hermitian and positive
        semidefinite, in numpy's Kronecker ordering.

    Raises
    ------
    RuntimeError
        If the solver stops with a status other than ``optimal`` or
        ``optimal_inaccurate``.
    cvxpy.error.SolverError
        If the solver fails outright. With ``closed_form=False`` and a pure three-level
        marginal this is the usual outcome (see the Notes).

    Notes
    -----
    A pure marginal leaves a single coupling. If :math:`\\rho_A = |\\psi\\rangle\\langle
    \\psi|`, the operator :math:`(I - |\\psi\\rangle\\langle\\psi|) \\otimes I` is positive
    and has zero expectation in any coupling, so the support of the coupling lies in
    :math:`|\\psi\\rangle \\otimes \\mathcal{H}_B` and the coupling is :math:`\\rho_A \\otimes
    \\rho_B`. The feasible set is one point and the minimum is its cost,
    :math:`\\mathrm{tr}(C\\,\\rho_A \\otimes \\rho_B)`. With the cost
    :math:`I - \\mathrm{SWAP}` and two pure states this is :math:`1 -
    |\\langle\\psi|\\phi\\rangle|^2`.

    When the pure marginal has dimension two or more, that single point has the rank of
    the other marginal, below :math:`d_A d_B`, so it sits on the boundary of the PSD
    cone, where an interior-point method has no interior to walk through. Measured on
    CLARABEL over ten random draws per case (``numpy.random.default_rng(0)``): on
    three-level systems the solver raised ``SolverError`` on nine draws for a pure state
    against a full-rank one, the remaining draw being off by 5.4e-4, and on seven draws
    for two pure states, the worst of the remaining three being off by 6.9e-4. The same
    counts were obtained at the default tolerance and at a gap of 1e-8, so the tight gap
    requested here is not the cause. On two-level systems the solver did not fail and
    its worst error over the ten draws was 8.8e-5. The closed form replaces all of this
    by a Kronecker product, exact to rounding error.

    A marginal is taken as pure when its spectrum equals :math:`(0, \\dots, 0, 1)` within
    1e-12, the test of the entropic solver
    (:func:`qot_course.quantum_ot.sinkhorn.quantum_sinkhorn_sdp`). A nearly pure
    marginal is not pure and goes to the solver. So does any pair in which one input
    is not a density matrix (Hermitian, unit trace, no negative eigenvalue, each
    within 1e-12): such a pair has no coupling, and the solver reports it as
    infeasible.

    The closed form is an elementary consequence of positivity; the SDP itself is the
    coupling problem of Cole, Eckstein, Friedland and Zyczkowski, Math. Phys. Anal.
    Geom. 26, 14 (2023), doi:10.1007/s11040-023-09456-7.

    Examples
    --------
    >>> import numpy as np
    >>> plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    >>> mixed = np.eye(2, dtype=complex) / 2
    >>> value, plan = quantum_ot_sdp(plus, mixed, quadratic_position_cost([0.0, 1.0]))
    >>> round(value, 12)
    0.5
    >>> bool(np.allclose(plan, np.kron(plus, mixed)))
    True
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    cost = np.asarray(cost, dtype=complex)
    d_a = rho_a.shape[0]
    d_b = rho_b.shape[0]

    if closed_form and _pure_marginal_fixes_the_coupling(rho_a, rho_b):
        # A pure marginal leaves one coupling only, the product (see the Notes).
        product = np.kron(rho_a, rho_b)
        return float(np.real(np.trace(cost @ product))), product

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
    # visible, not silenced). We accept that status below. Measured against closed
    # forms on two-level inputs, the returned value is within 1e-7 on the pairs the
    # course uses (|+><+| against I/2, |0> against |1>) and within 8.8e-5 on random
    # pure pairs; a pure three-level marginal makes the solver fail (see the Notes).
    chosen_solver = solver if solver is not None else "CLARABEL"
    if chosen_solver == "CLARABEL":
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10)
    else:
        problem.solve(solver=chosen_solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QOT SDP did not converge (status: {problem.status}).")
    return float(problem.value), np.asarray(plan.value, dtype=complex)


def quantum_wasserstein_squared_swap(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    solver: str | None = None,
    closed_form: bool = True,
) -> float:
    """Quantum-Wasserstein-:math:`^2` with SWAP cost :math:`I - \\mathrm{SWAP}`.

    Equivalent to :math:`\\mathrm{QOT}(\\rho_A, \\rho_B)` with :math:`C = I - \\mathrm{SWAP}`.
    ``solver`` and ``closed_form`` are passed to :func:`quantum_ot_sdp`: by default a
    pure marginal returns :math:`\\mathrm{tr}((I - \\mathrm{SWAP})\\,\\rho_A \\otimes \\rho_B)`
    without a solver call, and ``closed_form=False`` sends it to the solver.
    """
    if rho_a.shape != rho_b.shape:
        raise ValueError("SWAP cost requires equal-dimension subsystems.")
    d = rho_a.shape[0]
    value, _ = quantum_ot_sdp(
        rho_a, rho_b, swap_cost(d), solver=solver, closed_form=closed_form
    )
    return value
