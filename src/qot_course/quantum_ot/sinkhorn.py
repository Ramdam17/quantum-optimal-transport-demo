"""Quantum entropic optimal transport --- the operator-level Sinkhorn.

S14 lifts S10's entropic regularisation to density matrices:

.. math::
    \\rho^\\star_{AB,\\,\\varepsilon} = \\arg\\min_{\\rho_{AB} \\in \\Pi(\\rho_A, \\rho_B)}
    \\ \\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\, S(\\rho_{AB}),

with :math:`S(\\rho_{AB}) = -\\mathrm{tr}(\\rho_{AB}\\log\\rho_{AB})` (von Neumann entropy).
The **matrix-exponential Gibbs kernel** :math:`K = \\exp(-C/\\varepsilon)` plays the role
of the entrywise classical kernel.

**Quantum Amari bridge.** Since :math:`\\log K = -C/\\varepsilon`, one line of algebra gives

.. math::
    \\varepsilon\\,S_{\\mathrm{Umegaki}}(\\rho_{AB}\\,\\|\\,K)
    = \\varepsilon\\,\\mathrm{tr}(\\rho_{AB}\\,\\log\\rho_{AB})
    - \\varepsilon\\,\\mathrm{tr}(\\rho_{AB}\\,\\log K)
    = \\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\,S(\\rho_{AB}),

so the entropic QOT plan is the **Umegaki relative-entropy projection** of the
matrix-exponential Gibbs kernel onto the coupling polytope. The Sinkhorn iterations of
S10 are *iterative Bregman projections* under the KL divergence; their quantum analogue
is the operator Bregman iteration under the Umegaki divergence (Peyre, Chizat, Vialard,
Solomon, 2019; Feliciangeli, Gerolin, Portinale, 2023).

For pedagogical clarity we solve the entropic SDP directly via cvxpy's
:func:`cvxpy.von_neumann_entr` atom rather than running the explicit operator iteration.
The two approaches return the same plan when :math:`\\rho_A`, :math:`\\rho_B` and
:math:`C` commute, where both reduce to classical Sinkhorn. Commuting is enough for that, not
necessary: a separable cost :math:`C_A \\otimes I + I \\otimes C_B` also gives both the
same plan, the product state. Off these cases the two are different objects and their
plans can differ; see the notes of :func:`operator_sinkhorn`.

References: M. Cuturi, "Sinkhorn distances", NeurIPS (2013); G. Peyre, L. Chizat, F.-X.
Vialard, J. Solomon, "Quantum entropic regularization of matrix-valued optimal
transport", European J. Appl. Math. 30, 1079 (2019), doi:10.1017/S0956792517000274;
D. Feliciangeli, A. Gerolin, L. Portinale, "A non-commutative entropic optimal transport
approach to quantum composite systems at positive temperature", J. Funct. Anal. 285,
109963 (2023), doi:10.1016/j.jfa.2023.109963; D. Trevisan, "Quantum optimal transport:
an invitation", Boll. Unione Mat. Ital. 18, 347 (2025), doi:10.1007/s40574-024-00428-5;
cvxpy docs.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import scipy.linalg as spla

from qot_course.quantum.composite import partial_trace
from qot_course.quantum.density import von_neumann_entropy


def matrix_gibbs_kernel(cost: np.ndarray, epsilon: float) -> np.ndarray:
    """Matrix-exponential Gibbs kernel :math:`K = \\exp(-C/\\varepsilon)`.

    Quantum analogue of the entrywise classical Gibbs kernel
    :math:`K_{ij} = e^{-C_{ij}/\\varepsilon}` from S10. For commuting (diagonal) cost
    operators the two coincide; in general the matrix exponential adds operator-mixing.
    """
    return spla.expm(-np.asarray(cost, dtype=complex) / float(epsilon))


# The spectrum of a density matrix built from a normalised ket is (0, ..., 0, 1) up to
# rounding: over 20,000 random kets in dimensions 2, 3, 4 and 9, no eigenvalue was
# measured further than 1.4e-15 from that pattern. The threshold below leaves three
# orders of magnitude above that rounding level and still means "pure up to rounding",
# not "nearly pure".
_PURE_STATE_EIGENVALUE_ATOL = 1e-12


def _is_pure_state(rho: np.ndarray) -> bool:
    """Tell whether a density matrix is a pure state, up to floating-point rounding.

    When to use: to recognise the marginals for which the coupling is unique, before
    deciding whether a transport problem needs a solver at all.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian matrix, shape ``(d, d)``, complex. It is expected to be a density
        matrix (positive semidefinite, trace one) but is not assumed to be one.

    Returns
    -------
    bool
        ``True`` when the largest eigenvalue is within
        :data:`_PURE_STATE_EIGENVALUE_ATOL` of one and every other eigenvalue is within
        the same distance of zero. A matrix that is not a density matrix (a negative
        eigenvalue, a trace different from one, the zero matrix) is therefore never
        reported as pure.

    Examples
    --------
    >>> import numpy as np
    >>> _is_pure_state(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex))
    True
    >>> _is_pure_state(np.eye(2, dtype=complex) / 2)
    False
    >>> _is_pure_state(np.diag([1.2, -0.2]).astype(complex))
    False
    """
    eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    largest_is_one = abs(float(eigenvalues[-1]) - 1.0) <= _PURE_STATE_EIGENVALUE_ATOL
    others_vanish = bool(
        np.all(np.abs(eigenvalues[:-1]) <= _PURE_STATE_EIGENVALUE_ATOL)
    )
    return bool(largest_is_one and others_vanish)


# A density matrix built by normalising a ket or a positive matrix is Hermitian, of unit
# trace and positive up to rounding: over 40,000 random pure and full-rank states in
# dimensions 2, 3, 4 and 9 the trace was within 6.7e-16 of one, the anti-Hermitian part
# below 5.6e-17 and the smallest eigenvalue above -5.8e-16. The threshold leaves three
# orders of magnitude above that level.
_DENSITY_MATRIX_ATOL = 1e-12


def _is_density_matrix(rho: np.ndarray) -> bool:
    """Tell whether an array is a density matrix, up to floating-point rounding.

    When to use: before applying a formula that is only true for states, to leave any
    other input to the code path that rejects it.

    Parameters
    ----------
    rho : np.ndarray
        Any array. Only a square complex or real matrix of shape ``(d, d)`` can pass.

    Returns
    -------
    bool
        ``True`` when ``rho`` is square, equals its conjugate transpose, has trace one
        and has no eigenvalue below zero, each within :data:`_DENSITY_MATRIX_ATOL`.

    Examples
    --------
    >>> import numpy as np
    >>> _is_density_matrix(np.eye(2, dtype=complex) / 2)
    True
    >>> _is_density_matrix(np.eye(2, dtype=complex) / 4)
    False
    >>> _is_density_matrix(np.array([[0.5, 0.3], [0.0, 0.5]], dtype=complex))
    False
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False
    if np.max(np.abs(rho - rho.conj().T)) > _DENSITY_MATRIX_ATOL:
        return False
    if abs(float(np.real(np.trace(rho))) - 1.0) > _DENSITY_MATRIX_ATOL:
        return False
    return bool(np.linalg.eigvalsh(rho)[0] >= -_DENSITY_MATRIX_ATOL)


def _pure_marginal_fixes_the_coupling(rho_a: np.ndarray, rho_b: np.ndarray) -> bool:
    """Tell whether the product state is the only coupling of the two marginals.

    When to use: to decide whether a transport problem between ``rho_a`` and ``rho_b``
    can be answered by :math:`\\rho_A \\otimes \\rho_B` without a solver.

    Parameters
    ----------
    rho_a : np.ndarray
        First marginal, shape ``(d_a, d_a)``.
    rho_b : np.ndarray
        Second marginal, shape ``(d_b, d_b)``.

    Returns
    -------
    bool
        ``True`` when both arrays are density matrices and at least one is a pure
        state. A pure state beside a matrix that is not a state has no coupling at
        all, so both marginals are checked.

    Examples
    --------
    >>> import numpy as np
    >>> pure = np.diag([1.0, 0.0]).astype(complex)
    >>> _pure_marginal_fixes_the_coupling(pure, np.eye(2, dtype=complex) / 2)
    True
    >>> _pure_marginal_fixes_the_coupling(pure, np.eye(2, dtype=complex) / 4)
    False
    """
    if not (_is_density_matrix(rho_a) and _is_density_matrix(rho_b)):
        return False
    return _is_pure_state(rho_a) or _is_pure_state(rho_b)


def quantum_sinkhorn_sdp(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    solver: str | None = None,
) -> tuple[float, np.ndarray]:
    """Solve the von-Neumann-entropy-regularised quantum OT problem.

    Minimises :math:`\\mathrm{tr}(C\\,\\rho_{AB}) - \\varepsilon\\,S(\\rho_{AB})` over
    Hermitian PSD couplings with prescribed partial-trace marginals, and returns the
    optimal objective value together with the optimal plan.

    When to use: as the convex-solver reference for the entropic problem on small
    systems (two qubits or two qutrits). Use :func:`operator_sinkhorn` when you want
    the alternating iteration itself; the two return the same plan in the commuting
    case and different plans in general (see the notes of :func:`operator_sinkhorn`).

    Parameters
    ----------
    rho_a : np.ndarray
        Density matrix of the first marginal, shape ``(d_a, d_a)``, complex, trace one.
    rho_b : np.ndarray
        Density matrix of the second marginal, shape ``(d_b, d_b)``, complex, trace one.
    cost : np.ndarray
        Hermitian cost operator, shape ``(d_a * d_b, d_a * d_b)``, complex, in the
        Kronecker ordering ``A`` then ``B``.
    epsilon : float
        Regularisation strength :math:`\\varepsilon > 0`, in the units of ``cost`` per
        nat of entropy.
    solver : str, optional
        cvxpy solver name. Defaults to ``"CLARABEL"`` with a tightened gap tolerance.
        Ignored when a marginal is pure, because no solve takes place.

    Returns
    -------
    objective : float
        :math:`\\mathrm{tr}(C P) - \\varepsilon\\,S(P)` evaluated on the returned plan,
        with :math:`S` in nats and the convention :math:`0 \\log 0 = 0`.
    plan : np.ndarray
        Optimal coupling :math:`P`, shape ``(d_a * d_b, d_a * d_b)``, complex.

    Raises
    ------
    RuntimeError
        If the solver stops with a status other than ``"optimal"`` or
        ``"optimal_inaccurate"``.

    Notes
    -----
    cvxpy's :func:`cvxpy.von_neumann_entr` uses the natural logarithm, the convention
    of the classical Sinkhorn module, so the two values of :math:`\\varepsilon` compare
    directly.

    **Pure marginal: closed form, no solve.** If :math:`\\rho_A = |\\psi\\rangle\\langle
    \\psi|`, every coupling satisfies :math:`\\mathrm{tr}\\bigl[((I - |\\psi\\rangle
    \\langle\\psi|) \\otimes I)\\,\\rho_{AB}\\bigr] = \\mathrm{tr}\\bigl[(I - |\\psi\\rangle
    \\langle\\psi|)\\,\\rho_A\\bigr] = 0`. A positive semidefinite matrix with zero
    expectation on a projector is supported on the complement, so :math:`\\rho_{AB} =
    |\\psi\\rangle\\langle\\psi| \\otimes \\sigma`, and the second marginal forces
    :math:`\\sigma = \\rho_B`. The feasible set is the single point :math:`\\rho_A
    \\otimes \\rho_B`, whatever the cost and whatever :math:`\\varepsilon`; the same
    holds with the roles exchanged. The function returns that product and the objective
    :math:`\\mathrm{tr}(C\\,\\rho_A \\otimes \\rho_B) - \\varepsilon\\,(S(\\rho_A) +
    S(\\rho_B))` directly. A marginal counts as pure when its spectrum is
    :math:`(0, \\dots, 0, 1)` within :data:`_PURE_STATE_EIGENVALUE_ATOL`, which is the
    rounding level of a state built from a normalised ket. The product is returned
    only when both inputs are density matrices (Hermitian, unit trace, no negative
    eigenvalue, each within 1e-12): a pure state beside a matrix that is not a state
    has no coupling, so such a pair reaches the solver and fails there as before.

    This case is taken out of the solver because the unique coupling lies on the
    boundary of the PSD cone, where the interior-point method stops with status
    ``"optimal_inaccurate"``. Measured before the closed form was added: an objective
    off by 3e-4 to 5e-4 on :math:`|+\\rangle\\langle+|` against :math:`I/2` with the
    quadratic cost (:math:`\\varepsilon` from 0.05 to 5), and by 2e-2 to 3.5e-2 on a
    random pure qutrit against a random full-rank qutrit with a random Hermitian cost.

    A marginal that is nearly pure, or rank-deficient without being pure (rank two in
    dimension three, for instance), still goes to the solver and still meets that
    boundary. Measured with eigenvalues :math:`(1 - 10^{-10},\\ 10^{-10})` on the
    eigenvectors :math:`|+\\rangle, |-\\rangle` against :math:`I/2`, with the quadratic
    cost: the returned plan is 2.6e-4 away from the product, although positivity keeps
    every coupling within about :math:`\\sqrt{10^{-10}} = 10^{-5}` of it. For every
    solved case the objective is evaluated on the returned plan rather than read from
    ``problem.value``, which is ``inf`` there: the solver returns eigenvalues of order
    :math:`-10^{-7}` and cvxpy's own evaluation of the entropy at such a point is
    :math:`-\\infty`.

    Examples
    --------
    >>> import numpy as np
    >>> from qot_course.quantum_ot.sdp import quadratic_position_cost
    >>> plus = np.full((2, 2), 0.5, dtype=complex)
    >>> mixed = np.eye(2, dtype=complex) / 2
    >>> cost = quadratic_position_cost([0.0, 1.0])
    >>> value, plan = quantum_sinkhorn_sdp(plus, mixed, cost, epsilon=0.4)
    >>> bool(np.isclose(value, 0.5 - 0.4 * np.log(2.0), atol=1e-12))
    True
    >>> bool(np.allclose(plan, np.kron(plus, mixed), atol=1e-12))
    True
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    cost = np.asarray(cost, dtype=complex)
    d_a, d_b = rho_a.shape[0], rho_b.shape[0]

    if _pure_marginal_fixes_the_coupling(rho_a, rho_b):
        # A pure marginal leaves one coupling only, the product (see the Notes).
        plan_value = np.kron(rho_a, rho_b)
    else:
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

        # We request a tight gap (1e-9); CLARABEL may not certify it and can return
        # status "optimal_inaccurate" (with a UserWarning we leave visible, not
        # silenced). In practice the returned value is accurate to ~1e-7--1e-8,
        # confirmed against the entropic->SDP and Amari-bridge regression tests. We
        # accept that status below.
        chosen_solver = solver if solver is not None else "CLARABEL"
        if chosen_solver == "CLARABEL":
            problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9)
        else:
            problem.solve(solver=chosen_solver)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Quantum Sinkhorn SDP did not converge ({problem.status})."
            )
        plan_value = np.asarray(plan.value, dtype=complex)
    # Evaluate the objective on the plan itself: ``problem.value`` is ``inf`` whenever
    # a solved plan touches the boundary of the PSD cone (see the Notes).
    transport_value = float(np.real(np.trace(cost @ plan_value)))
    entropy_nats = von_neumann_entropy(plan_value, base=np.e)
    return transport_value - float(epsilon) * entropy_nats, plan_value


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


# ----------------------------------------------------------------------------- #
# Explicit operator (quantum) Sinkhorn iteration --- the matrix analogue of the
# alternating row/column rescaling of classical Sinkhorn (S10). Where
# ``quantum_sinkhorn_sdp`` above hands the whole entropic SDP to cvxpy, the routines
# below run the *iteration itself*: alternating positive-operator rescalings of the
# Gibbs kernel until the partial traces hit the prescribed marginals. This is operator
# scaling in the sense of Gurvits (2004).
# ----------------------------------------------------------------------------- #


# Floor for eigenvalues in Hermitian matrix powers. The kernel ``K = expm(-C/eps)`` is
# PSD but the inner products formed during the iteration can carry eigenvalues that dip
# to ~machine-epsilon; clipping keeps ``M^{-1/2}`` finite without perturbing the
# physically relevant (non-negligible) spectrum.
_EIGENVALUE_FLOOR = 1e-12


def _psd_pow(
    matrix: np.ndarray, power: float, eps: float = _EIGENVALUE_FLOOR
) -> np.ndarray:
    """Raise a Hermitian matrix to a real power via its eigendecomposition.

    Computes :math:`M^{p} = U \\Lambda^{p} U^{\\dagger}` where :math:`M = U \\Lambda
    U^{\\dagger}` is the eigendecomposition of the Hermitised input. Eigenvalues are
    floored at ``eps`` before exponentiation so that negative powers (``power < 0``)
    stay finite when ``matrix`` is rank-deficient or numerically singular.

    Parameters
    ----------
    matrix : np.ndarray
        Hermitian (or near-Hermitian) matrix, shape ``(d, d)``, complex. The input is
        symmetrised as ``0.5 * (M + M^H)`` before the eigendecomposition, so tiny
        anti-Hermitian numerical noise is discarded.
    power : float
        Real exponent (dimensionless). ``0.5`` gives the principal square root,
        ``-0.5`` the inverse square root.
    eps : float, optional
        Lower clip on the real eigenvalues (default :data:`_EIGENVALUE_FLOOR`).

    Returns
    -------
    np.ndarray
        ``matrix ** power``, shape ``(d, d)``, complex Hermitian.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=complex)
    >>> np.round(_psd_pow(M, 0.5).real, 6)
    array([[2., 0.],
           [0., 3.]])
    """
    vals, vecs = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    vals = np.clip(vals.real, eps, None)
    return (vecs * (vals**power)) @ vecs.conj().T


def _scaling_factor(M: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Hermitian PSD ``X`` solving the operator equation :math:`X M X = T`.

    The unique Hermitian PSD solution is

    .. math::
        X = M^{-1/2}\\,\\bigl(M^{1/2}\\,T\\,M^{1/2}\\bigr)^{1/2}\\,M^{-1/2},

    the building block of the operator-scaling step (Gurvits, 2004): given the current
    "marginal" :math:`M = \\mathrm{tr}_{\\bar{\\cdot}}\\!\\big[(I\\otimes Y)K(I\\otimes
    Y)\\big]` and a target reduced state :math:`T`, ``X`` is the positive rescaling that
    forces that partial trace to equal :math:`T`. In the commuting/diagonal case it
    collapses to the elementwise :math:`X_{ii} = \\sqrt{T_{ii}/M_{ii}}` of classical
    Sinkhorn.

    Parameters
    ----------
    M : np.ndarray
        Hermitian PSD operator, shape ``(d, d)``, complex.
    target : np.ndarray
        Hermitian PSD target (a reduced density matrix here), shape ``(d, d)``, complex.

    Returns
    -------
    np.ndarray
        Hermitian PSD ``X``, shape ``(d, d)``, complex, satisfying ``X M X = target``.
    """
    M_half = _psd_pow(M, 0.5)
    M_ihalf = _psd_pow(M, -0.5)
    inner = _psd_pow(M_half @ target @ M_half, 0.5)
    return M_ihalf @ inner @ M_ihalf


# Convergence / iteration-budget defaults for ``operator_sinkhorn``. Exposed as keyword
# arguments so callers (and notebook 04/07) can trade accuracy for speed.
_DEFAULT_N_ITER = 500
_DEFAULT_TOL = 1e-12


def operator_sinkhorn(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    cost: np.ndarray,
    epsilon: float,
    n_iter: int = _DEFAULT_N_ITER,
    tol: float = _DEFAULT_TOL,
) -> tuple[np.ndarray, dict]:
    """Run the explicit operator (quantum) Sinkhorn iteration for entropic QOT.

    Scales the matrix Gibbs kernel :math:`K = \\exp(-C/\\varepsilon)` by a product of
    positive operators :math:`X \\otimes Y` so that the bipartite plan
    :math:`P = (X\\otimes Y)\\,K\\,(X\\otimes Y)` has the prescribed partial traces
    :math:`\\mathrm{tr}_B(P) = \\rho_A` and :math:`\\mathrm{tr}_A(P) = \\rho_B`. Each
    half-step solves the operator equation :math:`X M X = \\rho` in closed form via
    :func:`_scaling_factor` (operator scaling; Gurvits, 2004). This is the matrix
    analogue of the alternating row/column rescaling of classical Sinkhorn (S10): the
    Gibbs kernel replaces the entrywise kernel, and the positive operators
    :math:`X, Y` replace the diagonal scaling vectors :math:`u, v`.

    Parameters
    ----------
    rho_a : np.ndarray
        Target marginal on subsystem :math:`A`, shape ``(d_a, d_a)``, a Hermitian PSD
        density matrix (unit trace, dimensionless).
    rho_b : np.ndarray
        Target marginal on subsystem :math:`B`, shape ``(d_b, d_b)``, Hermitian PSD.
    cost : np.ndarray
        Hermitian cost operator on :math:`\\mathcal{H}_A \\otimes \\mathcal{H}_B`, shape
        ``(d_a * d_b, d_a * d_b)``. Units of the cost set the units of ``epsilon``.
    epsilon : float
        Entropic regularisation strength (same units as ``cost``); larger
        :math:`\\varepsilon` smooths the plan toward the product :math:`\\rho_A \\otimes
        \\rho_B`. Uses the natural-log convention, matching :func:`quantum_sinkhorn_sdp`.
    n_iter : int, optional
        Maximum number of alternating sweeps (default :data:`_DEFAULT_N_ITER`).
    tol : float, optional
        Stop once the summed marginal residual ``‖tr_B P - ρ_A‖_F + ‖tr_A P - ρ_B‖_F``
        drops below this (default :data:`_DEFAULT_TOL`, Frobenius norm).

    Returns
    -------
    plan : np.ndarray
        The Hermitised coupling :math:`P`, shape ``(d_a * d_b, d_a * d_b)``, complex
        Hermitian PSD, with partial traces equal to ``rho_a`` and ``rho_b``.
    log : dict
        Diagnostics with keys ``"marginal_errors"`` (list of the per-iteration summed
        Frobenius residuals, monotone non-increasing in practice) and ``"n_iter"`` (the
        number of sweeps actually run).

    When to use
    -----------
    Use this when you want the **algorithm itself** --- the operator generalisation of
    Sinkhorn's matrix-scaling iteration --- e.g. to teach how entropic QOT is *computed*
    or to scale past the small dimensions where the SDP is comfortable. Use
    :func:`quantum_sinkhorn_sdp` instead when you want a convex-solver reference value
    for the entropic objective :math:`\\mathrm{tr}(C P) - \\varepsilon S(P)`: that route
    hands the whole von-Neumann-entropy SDP to cvxpy and is the ground truth this
    iteration is validated against.

    Notes
    -----
    **Relationship to the SDP --- honest caveat.** In the **commuting case** (e.g.
    diagonal ``rho_a``, ``rho_b`` and a diagonal cost, so :math:`[\\rho_A, \\rho_B,
    C]` all commute) this iteration and :func:`quantum_sinkhorn_sdp` both reduce to
    *classical* Sinkhorn and therefore return the **same** plan --- this is what
    ``test_operator_sinkhorn_equals_sdp_in_commuting_case`` verifies. In the general
    **non-commuting** case the two can differ. They need not: a separable cost
    :math:`C_A \\otimes I + I \\otimes C_B` has the product state as both answers.
    This fixed point is the operator-scaling
    solution (Georgiou & Pavon, 2015), of the form :math:`(X \\otimes Y) K (X \\otimes Y)`,
    whereas the SDP optimum is the Umegaki projection of :math:`K`, of the form
    :math:`\\exp(\\log K + U \\otimes I + I \\otimes V)`. Measured on a non-commuting
    qubit pair at :math:`\\varepsilon = 0.5` (cost :math:`(X \\otimes I - I \\otimes Z)^2`,
    the two random states of notebook ``04/07``, drawn with ``default_rng(1)``):
    Frobenius distance 0.0983 between the plans, entropic objective 0.6467 at this fixed
    point against 0.6305 at the SDP optimum. This routine is **not** guaranteed to
    reproduce the SDP plan off the commuting case --- treat it as the operator-scaling
    solution in its own right.

    The square-root rescaling ``X M X = ρ`` (rather than a one-sided ``X M = ρ``) keeps
    every iterate Hermitian PSD, which is what makes ``P`` a valid (sub-normalised)
    density operator throughout.

    References
    ----------
    L. Gurvits, "Classical complexity and quantum entanglement", J. Comput. System Sci.
    69, 448 (2004). doi:10.1016/j.jcss.2004.06.003 (operator scaling).
    T. T. Georgiou, M. Pavon, "Positive contraction mappings for classical and quantum
    Schrödinger systems", J. Math. Phys. 56, 033301 (2015). doi:10.1063/1.4915289.

    Examples
    --------
    >>> import numpy as np
    >>> from qot_course.quantum_ot.sdp import quadratic_position_cost
    >>> from qot_course.quantum.composite import partial_trace
    >>> rho_a = np.diag([0.6, 0.4]).astype(complex)
    >>> rho_b = np.diag([0.3, 0.7]).astype(complex)
    >>> C = quadratic_position_cost([0.0, 1.0])
    >>> P, log = operator_sinkhorn(rho_a, rho_b, C, epsilon=0.3)
    >>> bool(np.allclose(partial_trace(P, keep=[0], dims=[2, 2]), rho_a, atol=1e-8))
    True
    >>> bool(np.allclose(partial_trace(P, keep=[1], dims=[2, 2]), rho_b, atol=1e-8))
    True
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    d_a, d_b = rho_a.shape[0], rho_b.shape[0]
    K = matrix_gibbs_kernel(cost, epsilon)
    I_a, I_b = np.eye(d_a, dtype=complex), np.eye(d_b, dtype=complex)
    X, Y = I_a.copy(), I_b.copy()
    errors: list[float] = []
    P = K
    it = 0
    for it in range(n_iter):
        # Half-step A: freeze Y, solve for X so that tr_B[(I⊗Y) K (I⊗Y)] rescales to ρ_A.
        IY = np.kron(I_a, Y)
        M_a = partial_trace(IY @ K @ IY, keep=[0], dims=[d_a, d_b])
        X = _scaling_factor(M_a, rho_a)
        # Half-step B: freeze X, solve for Y so that tr_A[(X⊗I) K (X⊗I)] rescales to ρ_B.
        XI = np.kron(X, I_b)
        M_b = partial_trace(XI @ K @ XI, keep=[1], dims=[d_a, d_b])
        Y = _scaling_factor(M_b, rho_b)
        # Assemble the current plan and measure how far its marginals sit from the targets.
        XY = np.kron(X, Y)
        P = XY @ K @ XY
        err = float(
            np.linalg.norm(partial_trace(P, keep=[0], dims=[d_a, d_b]) - rho_a)
            + np.linalg.norm(partial_trace(P, keep=[1], dims=[d_a, d_b]) - rho_b)
        )
        errors.append(err)
        if err < tol:
            break
    return 0.5 * (P + P.conj().T), {"marginal_errors": errors, "n_iter": it + 1}
