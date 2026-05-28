import numpy as np
import ot
import pytest

from qot_course.quantum.composite import partial_trace, tensor
from qot_course.quantum.density import density_matrix, maximally_mixed
from qot_course.quantum.states import KET_0, KET_1, KET_PLUS, qubit_state
from qot_course.quantum_ot.sdp import (
    quadratic_position_cost,
    quantum_ot_sdp,
    quantum_wasserstein_squared_swap,
    swap_cost,
    swap_matrix,
)


# ----------------------------- SWAP basics --------------------------------- #
def test_swap_matrix_is_involution_and_hermitian():
    S = swap_matrix(3)
    np.testing.assert_allclose(S @ S, np.eye(9), atol=1e-12)
    np.testing.assert_allclose(S, S.conj().T, atol=1e-12)


def test_swap_acts_correctly_on_basis_states():
    S = swap_matrix(2)
    ket_01 = np.kron(KET_0, KET_1)
    ket_10 = np.kron(KET_1, KET_0)
    np.testing.assert_allclose(S @ ket_01, ket_10, atol=1e-12)
    np.testing.assert_allclose(S @ ket_10, ket_01, atol=1e-12)


# ----------------------------- Marginal feasibility ----------------------- #
def _check_marginals(plan, rho_a, rho_b, dims=(2, 2), atol=1e-5):
    rho_a_marginal = partial_trace(plan, keep=[0], dims=list(dims))
    rho_b_marginal = partial_trace(plan, keep=[1], dims=list(dims))
    np.testing.assert_allclose(rho_a_marginal, rho_a, atol=atol)
    np.testing.assert_allclose(rho_b_marginal, rho_b, atol=atol)


def test_sdp_returns_a_valid_coupling():
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    value, plan = quantum_ot_sdp(rho_a, rho_b, swap_cost(2))
    assert value >= -1e-7
    # Hermitian PSD
    np.testing.assert_allclose(plan, plan.conj().T, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(0.5 * (plan + plan.conj().T))) > -1e-6
    # Marginals
    _check_marginals(plan, rho_a, rho_b)


# ----------------------------- Identical states give zero ----------------- #
def test_qot_zero_for_identical_states_swap_cost():
    rho = density_matrix(KET_PLUS)
    assert quantum_wasserstein_squared_swap(rho, rho) == pytest.approx(0.0, abs=1e-6)


def test_qot_zero_for_identical_mixed_states_quadratic_cost():
    rho = maximally_mixed(2)
    value, _ = quantum_ot_sdp(rho, rho, quadratic_position_cost([0.0, 1.0]))
    assert value == pytest.approx(0.0, abs=1e-6)


# ----------------------------- Diagonal collapse: QOT = classical OT ----- #
def test_quadratic_qot_on_diagonal_states_matches_classical_w2_squared():
    """Diagonal-collapse principle (S12): for diagonal rho_A, rho_B in the X eigenbasis,
    the quadratic-cost QOT equals the classical W_2^2 on the diagonals."""
    p = np.array([0.6, 0.4])
    q = np.array([0.1, 0.9])
    rho_a = np.diag(p).astype(complex)
    rho_b = np.diag(q).astype(complex)
    positions = np.array([0.0, 1.0])
    qot_value, _ = quantum_ot_sdp(
        rho_a, rho_b, quadratic_position_cost(positions)
    )
    cost_classical = (positions.reshape(-1, 1) - positions.reshape(1, -1)) ** 2
    classical_w2_sq = float(ot.emd2(p, q, cost_classical))
    assert qot_value == pytest.approx(classical_w2_sq, abs=1e-5)


# ----------------------------- Pure-state SWAP-cost identity -------------- #
def test_swap_qot_on_pure_states_equals_one_minus_squared_overlap():
    """SDP returns 1 - |<psi|phi>|^2 for pure states (Cole et al., 2023).

    Note: pure-state marginals are rank-1 constraints --- a degenerate case for
    interior-point SDP solvers. Tolerance ~1e-3 absolute is realistic on this
    boundary of the feasible cone; the closed-form identity itself is exact.
    """
    rng = np.random.default_rng(0)
    for _ in range(4):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = psi / np.linalg.norm(psi)
        phi = rng.normal(size=2) + 1j * rng.normal(size=2)
        phi = phi / np.linalg.norm(phi)
        rho_a = density_matrix(psi)
        rho_b = density_matrix(phi)
        expected = 1.0 - abs(np.vdot(psi, phi)) ** 2
        actual = quantum_wasserstein_squared_swap(rho_a, rho_b)
        assert actual == pytest.approx(expected, abs=1e-3)


def test_swap_qot_on_orthogonal_pure_states_equals_one():
    value = quantum_wasserstein_squared_swap(
        density_matrix(KET_0), density_matrix(KET_1)
    )
    assert value == pytest.approx(1.0, abs=1e-5)


# ----------------------------- Plus vs I/2 punchline ---------------------- #
def test_quadratic_qot_distinguishes_plus_from_maximally_mixed():
    """Classical W_2 on Z-diagonals is 0; the SDP must return > 0."""
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    qot_value, _ = quantum_ot_sdp(
        rho_a, rho_b, quadratic_position_cost([0.0, 1.0])
    )
    assert qot_value > 1e-4
    # And classical OT on diagonals would give 0.
    diag_a = np.diag(rho_a).real.copy()
    diag_b = np.diag(rho_b).real.copy()
    cost = (np.arange(2).reshape(-1, 1) - np.arange(2).reshape(1, -1)) ** 2.0
    assert ot.emd2(diag_a, diag_b, cost) == pytest.approx(0.0, abs=1e-9)


# ----------------------------- SWAP symmetry ------------------------------ #
def test_swap_qot_symmetric():
    rho_a = density_matrix(KET_PLUS)
    rho_b = density_matrix(qubit_state(theta=np.pi / 2, phi=np.pi / 2))  # |+i>
    forward = quantum_wasserstein_squared_swap(rho_a, rho_b)
    backward = quantum_wasserstein_squared_swap(rho_b, rho_a)
    assert forward == pytest.approx(backward, abs=1e-5)


# ----------------------------- Product-coupling sanity bound -------------- #
def test_qot_value_at_or_below_product_coupling_cost():
    """The independent coupling rho_A otimes rho_B is feasible -- the SDP optimum
    must be at most its cost (a useful sanity bound)."""
    rho_a = density_matrix(KET_PLUS)
    rho_b = maximally_mixed(2)
    cost_op = quadratic_position_cost([0.0, 1.0])
    qot_value, _ = quantum_ot_sdp(rho_a, rho_b, cost_op)
    product_coupling = tensor(rho_a, rho_b)
    independent_cost = float(np.real(np.trace(cost_op @ product_coupling)))
    assert qot_value <= independent_cost + 1e-5
