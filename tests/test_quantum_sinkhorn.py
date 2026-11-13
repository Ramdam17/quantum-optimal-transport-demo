import warnings

import cvxpy as cp
import numpy as np
import pytest

from qot_course.quantum.composite import partial_trace, tensor
from qot_course.quantum.density import (
    density_matrix,
    maximally_mixed,
    von_neumann_entropy,
)
from qot_course.quantum.states import KET_PLUS
from qot_course.quantum_ot.sdp import (
    quadratic_position_cost,
    quantum_ot_sdp,
)
from qot_course.quantum_ot.sinkhorn import (
    _is_density_matrix,
    _is_pure_state,
    matrix_gibbs_kernel,
    operator_sinkhorn,
    quantum_sinkhorn_cost,
    quantum_sinkhorn_sdp,
    umegaki_kl_to_kernel,
)
from tests.quantum_ot_helpers import (
    count_solver_calls,
    matrices_that_are_not_states,
    random_full_rank_state,
    random_hermitian,
    random_ket,
)


# ----------------------------- Matrix-exponential Gibbs kernel ----------- #
def test_matrix_gibbs_kernel_diagonal_cost_matches_entrywise_exp():
    """For diagonal C, expm(-C/eps) is just the entrywise exp on the diagonal."""
    C = np.diag([0.0, 1.0, 2.0, 0.0]).astype(complex)
    K = matrix_gibbs_kernel(C, epsilon=0.5)
    np.testing.assert_allclose(K, np.diag(np.exp(-np.diag(C) / 0.5)), atol=1e-10)


def test_matrix_gibbs_kernel_hermitian_and_positive_definite():
    """For Hermitian C, K = expm(-C/eps) is Hermitian PSD."""
    C = quadratic_position_cost([0.0, 1.0])
    K = matrix_gibbs_kernel(C, epsilon=0.5)
    np.testing.assert_allclose(K, K.conj().T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(0.5 * (K + K.conj().T))) > 0.0


# ----------------------------- Marginals match --------------------------- #
def test_quantum_sinkhorn_plan_satisfies_marginals():
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    _, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=0.5)
    a_back = partial_trace(plan, keep=[0], dims=[2, 2])
    b_back = partial_trace(plan, keep=[1], dims=[2, 2])
    np.testing.assert_allclose(a_back, rho_a, atol=1e-5)
    np.testing.assert_allclose(b_back, rho_b, atol=1e-5)


# ----------------------------- ε limits --------------------------------- #
def test_quantum_sinkhorn_approaches_sdp_at_small_epsilon():
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    # |+><+| is pure, so both functions would return the same product formula.
    # closed_form=False makes the reference the solver's own value.
    sdp_value, _ = quantum_ot_sdp(rho_a, rho_b, C, closed_form=False)
    small_eps_value = quantum_sinkhorn_cost(rho_a, rho_b, C, epsilon=0.02)
    # The transport cost at small eps should be close to the SDP optimum.
    assert small_eps_value == pytest.approx(sdp_value, abs=0.05)


def test_quantum_sinkhorn_approaches_product_at_large_epsilon():
    """At large ε the entropic optimum is the product coupling ρ_A ⊗ ρ_B."""
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    _, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=200.0)
    product = tensor(rho_a, rho_b)
    np.testing.assert_allclose(plan, product, atol=1e-3)


# ----------------------------- Amari quantum bridge --------------------- #
def _von_neumann_entropy_nats(state: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (state + state.conj().T))
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log(vals)))


def test_amari_quantum_bridge_identity_holds_for_every_state():
    """ε * S_Umegaki(P || K) = tr(C P) - ε * S(P) for any state P, since log K = -C/ε.

    The identity is algebra, so it checks ``umegaki_kl_to_kernel`` and
    ``matrix_gibbs_kernel`` against the two terms written out, on a state that no
    solver produced. The optimum is checked by the two tests that follow.
    """
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.3
    K = matrix_gibbs_kernel(C, eps)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    state = raw @ raw.conj().T
    state /= np.trace(state).real

    lhs = eps * umegaki_kl_to_kernel(state, K)
    rhs = float(np.real(np.trace(C @ state))) - eps * _von_neumann_entropy_nats(state)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_entropic_optimum_with_a_pure_marginal_has_its_closed_form_value():
    """With ρ_A pure the only coupling is ρ_A ⊗ ρ_B, so the optimum is known by hand.

    For |+><+| and I/2 the product has cost tr(C P) = 1/2 and entropy ln 2, hence
    ε * S_Umegaki(P_eps || K) = 1/2 - ε ln 2.
    """
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.3
    _, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, eps)
    K = matrix_gibbs_kernel(C, eps)
    np.testing.assert_allclose(plan, tensor(rho_a, rho_b), atol=1e-6)
    assert eps * umegaki_kl_to_kernel(plan, K) == pytest.approx(
        0.5 - eps * np.log(2.0), abs=1e-6
    )


def test_quantum_sinkhorn_plan_minimises_umegaki_kl_to_kernel():
    """The entropic optimum minimises S_Umegaki(P || K) over the couplings.

    Two mixed marginals that do not commute leave room to move: adding δ σ_i ⊗ σ_j,
    with σ_i and σ_j Pauli matrices, changes neither marginal because both factors are
    traceless. Each of the nine directions, in both senses and at two step sizes, must
    increase the divergence. A pure marginal would leave no direction to try.
    """
    identity = np.eye(2, dtype=complex)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    x, y, z = paulis
    rho_a = 0.5 * (identity + 0.5 * x + 0.3 * z)
    rho_b = 0.5 * (identity + 0.4 * z - 0.2 * y)
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.4
    _, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, eps)
    K = matrix_gibbs_kernel(C, eps)
    kl_star = umegaki_kl_to_kernel(plan, K)

    # The plan is a coupling, and it is not the product: there is something to minimise.
    a_star = partial_trace(plan, keep=[0], dims=[2, 2])
    b_star = partial_trace(plan, keep=[1], dims=[2, 2])
    np.testing.assert_allclose(a_star, rho_a, atol=1e-8)
    np.testing.assert_allclose(b_star, rho_b, atol=1e-8)
    assert umegaki_kl_to_kernel(tensor(rho_a, rho_b), K) > kl_star + 1e-2

    tried = 0
    for sigma_a in paulis:
        for sigma_b in paulis:
            for delta in (-0.01, -0.005, 0.005, 0.01):
                perturbed = plan + delta * np.kron(sigma_a, sigma_b)
                assert np.linalg.eigvalsh(perturbed).min() > 0.0  # still a state
                assert umegaki_kl_to_kernel(perturbed, K) > kl_star + 1e-6
                tried += 1
    assert tried == 36


def test_umegaki_kl_to_kernel_zero_when_plan_equals_kernel():
    K = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=complex)
    assert umegaki_kl_to_kernel(K, K) == pytest.approx(0.0, abs=1e-9)


# ----------------------------- Punchline preserved ---------------------- #
def test_entropic_qot_still_distinguishes_plus_from_mixed():
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    # Even with entropy regularisation, the transport cost is positive.
    cost_value = quantum_sinkhorn_cost(rho_a, rho_b, C, epsilon=0.3)
    assert cost_value > 1e-3


# ----------------------------- Operator (quantum) Sinkhorn iteration ----- #
def test_operator_sinkhorn_matches_marginals():
    rho_a = np.diag([0.6, 0.4]).astype(complex)
    rho_b = np.diag([0.3, 0.7]).astype(complex)
    C = quadratic_position_cost([0.0, 1.0])
    P, log = operator_sinkhorn(rho_a, rho_b, C, epsilon=0.3)
    np.testing.assert_allclose(
        partial_trace(P, keep=[0], dims=[2, 2]), rho_a, atol=1e-8
    )
    np.testing.assert_allclose(
        partial_trace(P, keep=[1], dims=[2, 2]), rho_b, atol=1e-8
    )
    assert log["marginal_errors"][-1] < 1e-8
    assert log["marginal_errors"][-1] <= log["marginal_errors"][0]


def test_operator_sinkhorn_equals_sdp_in_commuting_case():
    # Diagonal states + diagonal cost commute -> operator scaling == von-Neumann SDP plan.
    rho_a = np.diag([0.6, 0.4]).astype(complex)
    rho_b = np.diag([0.3, 0.7]).astype(complex)
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.3
    P_iter, _ = operator_sinkhorn(rho_a, rho_b, C, epsilon=eps)
    _, P_sdp = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=eps)
    np.testing.assert_allclose(P_iter, P_sdp, atol=1e-6)


# ----------------------------- Returned objective value ------------------ #
# A pure marginal admits exactly one coupling, the product state, so the function
# returns it in closed form without calling the solver. The bound is the rounding error
# of a Kronecker product and one eigendecomposition; the solver it replaces was off by
# 3e-4 to 5e-4 on the qubit pair below and by 2e-2 to 3.5e-2 on the qutrit case.
PURE_MARGINAL_EXACT_ATOL = 1e-12


@pytest.mark.parametrize("pure_side", ["a", "b"])
def test_quantum_sinkhorn_sdp_is_exact_for_a_pure_marginal(pure_side):
    """With a pure marginal the returned objective is tr(C rho_A x rho_B) - eps ln 2.

    Regression test twice over: the value used to come back as ``inf``, then as a number
    accurate to about 5e-4 only, because the plan sits on the boundary of the PSD cone.
    """
    plus = density_matrix(KET_PLUS)
    mixed = maximally_mixed(2)
    rho_a, rho_b = (plus, mixed) if pure_side == "a" else (mixed, plus)
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.4
    with warnings.catch_warnings():
        # The closed form must not go through the solver, so no accuracy warning either.
        warnings.simplefilter("error")
        value, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=eps)
    # Unique coupling = product; tr(C * product) = 1/2 and S(product) = ln 2 nats.
    expected = 0.5 - eps * np.log(2.0)
    assert value == pytest.approx(expected, abs=PURE_MARGINAL_EXACT_ATOL)
    np.testing.assert_allclose(
        plan, tensor(rho_a, rho_b), atol=PURE_MARGINAL_EXACT_ATOL
    )


@pytest.mark.parametrize("pure_side", ["a", "b"])
def test_quantum_sinkhorn_sdp_is_exact_for_a_pure_qutrit_marginal(pure_side):
    """Pure qutrit against a full-rank qutrit, generic cost: value = tr(C P) - eps S(mixed)."""
    rng = np.random.default_rng(0)
    pure = density_matrix(random_ket(rng, 3))
    full_rank = random_full_rank_state(rng, 3)
    rho_a, rho_b = (pure, full_rank) if pure_side == "a" else (full_rank, pure)
    C = random_hermitian(rng, 9)
    eps = 0.3
    value, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=eps)
    product = tensor(rho_a, rho_b)
    # S(rho_A x rho_B) = S(rho_A) + S(rho_B), and the entropy of the pure side is zero.
    expected = float(np.real(np.trace(C @ product))) - eps * von_neumann_entropy(
        full_rank, base=np.e
    )
    np.testing.assert_allclose(plan, product, atol=PURE_MARGINAL_EXACT_ATOL)
    assert value == pytest.approx(expected, abs=PURE_MARGINAL_EXACT_ATOL)


def test_quantum_sinkhorn_sdp_is_exact_when_both_marginals_are_pure():
    """Two pure marginals: the plan is |psi><psi| x |phi><phi| and its entropy is zero."""
    rng = np.random.default_rng(1)
    rho_a = density_matrix(random_ket(rng, 2))
    rho_b = density_matrix(random_ket(rng, 2))
    C = random_hermitian(rng, 4)
    value, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=0.7)
    product = tensor(rho_a, rho_b)
    np.testing.assert_allclose(plan, product, atol=PURE_MARGINAL_EXACT_ATOL)
    assert value == pytest.approx(
        float(np.real(np.trace(C @ product))), abs=PURE_MARGINAL_EXACT_ATOL
    )


# Spectra on each side of the purity threshold of 1e-12. The second eigenvalue of a
# state built from a normalised ket is of order 1e-16; 1e-13 is still rounding, 1e-9 is
# a mixed state whose couplings are not unique.
@pytest.mark.parametrize(
    ("second_eigenvalue", "expected_solver_calls"),
    [(0.0, 0), (1e-13, 0), (1e-9, 1), (1e-6, 1)],
)
def test_quantum_sinkhorn_sdp_skips_the_solver_only_for_a_pure_marginal(
    monkeypatch, second_eigenvalue, expected_solver_calls
):
    """The closed form is used for a marginal that is pure up to rounding, and for no other."""
    calls = count_solver_calls(monkeypatch)
    rho_a = np.diag([1.0 - second_eigenvalue, second_eigenvalue]).astype(complex)
    C = quadratic_position_cost([0.0, 1.0])
    quantum_sinkhorn_sdp(rho_a, maximally_mixed(2), C, epsilon=0.4)
    assert len(calls) == expected_solver_calls


@pytest.mark.parametrize(
    "not_a_state",
    [
        # Trace one, largest eigenvalue above one, one negative eigenvalue.
        np.diag([1.0 + 1e-9, -1e-9]),
        np.diag([1.2, -0.2]),
        # Largest eigenvalue one, but the others do not vanish.
        np.diag([1.0, 0.3, -0.3]),
        # A pure state multiplied by two: not normalised.
        2.0 * np.full((2, 2), 0.5),
        # The zero matrix.
        np.zeros((2, 2)),
    ],
)
def test_is_pure_state_rejects_matrices_that_are_not_pure_states(not_a_state):
    """Only the spectrum (0, ..., 0, 1) counts as pure; invalid inputs go to the solver."""
    assert not _is_pure_state(not_a_state.astype(complex))


def test_is_pure_state_accepts_pure_states_of_any_dimension():
    rng = np.random.default_rng(2)
    for dim in (1, 2, 3, 9):
        assert _is_pure_state(density_matrix(random_ket(rng, dim)))
    assert not _is_pure_state(maximally_mixed(2))


def test_quantum_sinkhorn_sdp_objective_matches_its_plan():
    """The returned value is tr(C P) - eps S(P) evaluated on the returned plan."""
    rho_a = np.diag([0.7, 0.3]).astype(complex)
    rho_b = maximally_mixed(2)
    C = quadratic_position_cost([0.0, 1.0])
    eps = 0.4
    value, plan = quantum_sinkhorn_sdp(rho_a, rho_b, C, epsilon=eps)
    transport = float(np.real(np.trace(C @ plan)))
    entropy_nats = von_neumann_entropy(plan, base=np.e)
    assert value == pytest.approx(transport - eps * entropy_nats, abs=1e-9)


# ----------------------------- The closed form needs two density matrices - #
@pytest.mark.parametrize("name", sorted(matrices_that_are_not_states()))
def test_is_density_matrix_rejects_matrices_that_are_not_states(name):
    assert not _is_density_matrix(matrices_that_are_not_states()[name])


def test_is_density_matrix_rejects_arrays_that_are_not_square_matrices():
    assert not _is_density_matrix(np.ones((2, 3), dtype=complex) / 6)
    assert not _is_density_matrix(np.array([1.0, 0.0], dtype=complex))


def test_is_density_matrix_accepts_pure_and_mixed_states_of_any_dimension():
    rng = np.random.default_rng(21)
    for dim in (1, 2, 3, 9):
        assert _is_density_matrix(density_matrix(random_ket(rng, dim)))
        assert _is_density_matrix(random_full_rank_state(rng, dim))
    assert _is_density_matrix(np.diag([0.5, 0.3, 0.2]).astype(complex))


@pytest.mark.parametrize("name", sorted(matrices_that_are_not_states()))
@pytest.mark.parametrize("pure_side", ["a", "b"])
def test_quantum_sinkhorn_sdp_sends_an_invalid_marginal_to_the_solver(
    monkeypatch, name, pure_side
):
    """A pure state beside a matrix that is not a state has no coupling at all.

    The product formula must not be applied: the input goes to the solver, which
    refuses it.
    """
    calls = count_solver_calls(monkeypatch)
    pure = np.diag([1.0, 0.0]).astype(complex)
    invalid = matrices_that_are_not_states()[name]
    rho_a, rho_b = (pure, invalid) if pure_side == "a" else (invalid, pure)
    cost = quadratic_position_cost([0.0, 1.0])
    with pytest.raises((RuntimeError, cp.error.SolverError, cp.error.DCPError)):
        quantum_sinkhorn_sdp(rho_a, rho_b, cost, epsilon=0.4)
    assert len(calls) == 1


def test_quantum_sinkhorn_sdp_sends_a_non_hermitian_pure_looking_marginal_to_the_solver(
    monkeypatch,
):
    """A matrix whose Hermitian part is pure is not a state when it is not Hermitian."""
    calls = count_solver_calls(monkeypatch)
    not_hermitian = np.array([[1.0, 0.3], [-0.3, 0.0]], dtype=complex)
    cost = quadratic_position_cost([0.0, 1.0])
    with pytest.raises((RuntimeError, cp.error.SolverError, cp.error.DCPError)):
        quantum_sinkhorn_sdp(not_hermitian, maximally_mixed(2), cost, epsilon=0.4)
    assert len(calls) == 1
