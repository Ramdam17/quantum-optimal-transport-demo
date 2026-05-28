import numpy as np
import pytest

from qot_course.quantum.composite import bell_state, tensor
from qot_course.quantum.density import density_matrix, maximally_mixed
from qot_course.quantum.states import KET_0, KET_1, KET_PLUS, qubit_state
from qot_course.quantum_ot.preview import (
    commutativity_norm,
    commutator,
    diagonal_in_computational_basis,
    same_diagonal,
)


# ----------------------------- Commutator basics ---------------------------- #
def test_commutator_antisymmetric():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    np.testing.assert_allclose(commutator(A, B), -commutator(B, A), atol=1e-12)


def test_commutator_of_identity_with_anything_is_zero():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    np.testing.assert_allclose(commutator(np.eye(4), A), 0.0, atol=1e-12)


def test_commutativity_norm_zero_iff_commuting():
    # Two diagonal matrices commute.
    D0 = np.diag([1.0, 2.0, 3.0])
    D1 = np.diag([0.5, 0.1, 0.9])
    assert commutativity_norm(D0, D1) == pytest.approx(0.0, abs=1e-12)
    # Two Pauli matrices do not commute.
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    assert commutativity_norm(X, Y) > 0.0


# ----------------------------- Plus-vs-mixed canonical example ------------- #
def test_plus_state_and_maximally_mixed_have_same_z_diagonal():
    plus = density_matrix(KET_PLUS)
    mixed = maximally_mixed(2)
    np.testing.assert_allclose(
        diagonal_in_computational_basis(plus), [0.5, 0.5]
    )
    np.testing.assert_allclose(
        diagonal_in_computational_basis(mixed), [0.5, 0.5]
    )
    assert same_diagonal(plus, mixed)


def test_plus_x_and_plus_y_have_same_z_diagonal_and_do_not_commute():
    plus_x = density_matrix(KET_PLUS)
    plus_y = density_matrix(qubit_state(theta=np.pi / 2, phi=np.pi / 2))  # |+i>
    assert same_diagonal(plus_x, plus_y)
    assert commutativity_norm(plus_x, plus_y) > 0.0


# ----------------------------- Same-diagonal predicate --------------------- #
def test_same_diagonal_negative_case():
    assert not same_diagonal(density_matrix(KET_0), density_matrix(KET_1))


def test_diagonal_in_computational_basis_is_real():
    # Even for matrices with complex off-diagonals, the diagonal of a Hermitian
    # density matrix is real.
    plus_y_state = qubit_state(theta=np.pi / 2, phi=np.pi / 2)
    plus_y = density_matrix(plus_y_state)
    diag = diagonal_in_computational_basis(plus_y)
    assert np.iscomplexobj(diag) is False or np.all(np.isreal(diag))


# ----------------------------- Bell state non-commutativity ---------------- #
def test_bell_and_overlapping_product_dont_commute():
    # Note: a pure state orthogonal to the Bell state (e.g. |+>|->) gives orthogonal
    # projectors, which commute. We pick a product state that overlaps with the Bell
    # state to expose the non-commutativity.
    rho_bell = density_matrix(bell_state())
    rho_product = density_matrix(tensor(KET_0, KET_0))
    assert commutativity_norm(rho_bell, rho_product) > 0.0
