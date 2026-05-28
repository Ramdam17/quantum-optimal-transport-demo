import numpy as np
import pytest

from qot_course.infotheory.quantum import (
    bures_distance,
    quantum_conditional_entropy,
    quantum_mutual_information,
    quantum_relative_entropy,
)
from qot_course.quantum.composite import (
    apply_channel,
    bell_state,
    depolarizing_channel,
    tensor,
)
from qot_course.quantum.density import (
    density_matrix,
    maximally_mixed,
    von_neumann_entropy,
)
from qot_course.quantum.states import KET_0, KET_1, KET_MINUS, KET_PLUS


# ---------------------------- Umegaki relative entropy --------------------- #
def test_quantum_relative_entropy_zero_iff_equal_and_nonnegative():
    rho = density_matrix(KET_PLUS)
    sigma = maximally_mixed(2)
    assert quantum_relative_entropy(rho, rho) == pytest.approx(0.0, abs=1e-9)
    assert quantum_relative_entropy(rho, sigma) > 0.0


def test_quantum_relative_entropy_to_maximally_mixed_equals_log_d_minus_entropy():
    # Closed form: S(rho || I/d) = log_2(d) - S(rho).
    d = 2
    for state in (KET_0, KET_PLUS, KET_MINUS):
        rho = density_matrix(state)
        expected = np.log2(d) - von_neumann_entropy(rho)
        assert quantum_relative_entropy(rho, maximally_mixed(d)) == pytest.approx(
            expected, abs=1e-9
        )


def test_quantum_relative_entropy_diverges_on_support_mismatch():
    # supp(|0><0|) = span(|0>) is disjoint from supp(|1><1|) -> +inf.
    assert np.isinf(
        quantum_relative_entropy(density_matrix(KET_0), density_matrix(KET_1))
    )


def test_quantum_relative_entropy_distinguishes_plus_from_max_mixed_classical_kl_cannot():
    # |+> and I/2 have *identical* diagonals in the Z basis, so classical KL on the
    # diagonals is 0 --- but Umegaki sees the coherence and returns 1 bit.
    rho = density_matrix(KET_PLUS)
    sigma = maximally_mixed(2)
    assert quantum_relative_entropy(rho, sigma) == pytest.approx(1.0, abs=1e-9)


# ---------------------------- Quantum mutual information ------------------ #
def test_bell_state_has_two_bits_of_quantum_mutual_information():
    rho = density_matrix(bell_state())
    assert quantum_mutual_information(rho, dims=[2, 2]) == pytest.approx(2.0)


def test_product_state_has_zero_mutual_information():
    rho = density_matrix(tensor(KET_0, KET_PLUS))
    assert quantum_mutual_information(rho, dims=[2, 2]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_mutual_information_nonnegative_on_random_bipartite_state():
    rng = np.random.default_rng(0)
    # Random pure state on 2 qubits.
    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    psi /= np.linalg.norm(psi)
    rho = density_matrix(psi)
    assert quantum_mutual_information(rho, dims=[2, 2]) >= -1e-12


# ---------------------------- Negative conditional entropy ---------------- #
def test_bell_state_has_negative_conditional_entropy():
    # The signature quantum effect: S(A|B) = -1 bit for the Bell pair.
    rho = density_matrix(bell_state())
    assert quantum_conditional_entropy(rho, dims=[2, 2]) == pytest.approx(-1.0)


def test_product_state_has_zero_conditional_entropy_in_pure_case():
    rho = density_matrix(tensor(KET_0, KET_0))
    assert quantum_conditional_entropy(rho, dims=[2, 2]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_strong_depolarization_makes_conditional_entropy_positive():
    # Push a Bell state through hard depolarization on qubit A:
    # the entanglement breaks and S(A|B) becomes >= 0 (classical regime).
    rho_bell = density_matrix(bell_state())
    kraus_a = depolarizing_channel(0.95)  # near-maximal noise
    # Apply on the first qubit only by tensoring with the identity on B.
    identity_b = np.eye(2, dtype=complex)
    kraus_ab = [np.kron(k, identity_b) for k in kraus_a]
    rho_noisy = apply_channel(rho_bell, kraus_ab)
    assert quantum_conditional_entropy(rho_noisy, dims=[2, 2]) > 0.0


# ---------------------------- Bures distance ------------------------------ #
def test_bures_distance_identical_is_zero():
    # Allow ~1e-7 numerical floor from the sqrtm chain inside `fidelity`.
    rho = density_matrix(KET_PLUS)
    assert bures_distance(rho, rho) == pytest.approx(0.0, abs=1e-6)


def test_bures_distance_orthogonal_pure_states_is_sqrt2():
    # F = 0 for orthogonal pure states -> d_B = sqrt(2(1 - 0)) = sqrt(2).
    d = bures_distance(density_matrix(KET_0), density_matrix(KET_1))
    assert d == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_bures_distance_symmetric():
    rho = density_matrix(KET_PLUS)
    sigma = maximally_mixed(2)
    assert bures_distance(rho, sigma) == pytest.approx(bures_distance(sigma, rho))


def test_bures_separates_plus_from_max_mixed_with_same_diagonal():
    # Classical Fisher--Rao on the (identical) diagonals would give 0; Bures sees
    # the off-diagonal coherence of |+> and returns a strictly positive distance.
    d = bures_distance(density_matrix(KET_PLUS), maximally_mixed(2))
    assert d > 0.0
