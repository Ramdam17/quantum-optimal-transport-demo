"""Check each entry of the course synthesis against a value worked out by hand."""

import numpy as np
import pytest

from qot_course.quantum_ot.synthesis import course_greatest_hits


@pytest.fixture(scope="module")
def hits():
    return course_greatest_hits()


# ----------------------------- Module 02: information ------------------- #
def test_s5_entropy_of_fair_coin_is_one_bit(hits):
    assert hits["02: H(fair coin) [bits]"] == pytest.approx(1.0)


def test_s5_kl_matches_its_two_terms(hits):
    # D(p || q) for p = (0.7, 0.3) and q = (0.4, 0.6), written out term by term.
    by_hand = 0.7 * np.log2(0.7 / 0.4) + 0.3 * np.log2(0.3 / 0.6)
    assert hits["02: D(p || q) > 0 [bits]"] == pytest.approx(by_hand)


def test_s5_mutual_information_endpoints(hits):
    assert hits["02: I(independent) [bits]"] == pytest.approx(0.0, abs=1e-12)
    assert hits["02: I(correlated) [bits]"] == pytest.approx(1.0)


# ----------------------------- Module 02: geometry ---------------------- #
def test_s6_mixture_midpoint_bimodal_w2_midpoint_unimodal(hits):
    # The result of 02/09: mixture-midpoint mass in the middle ≈ 0; W2-midpoint mass ≈ 1.
    assert hits["02: mixture midpoint mass in middle"] < 0.05
    assert hits["02: W2 midpoint mass in middle"] > 0.95


def test_s6_fisher_rao_distance_is_twice_the_bhattacharyya_angle(hits):
    # d_FR(p, q) = 2 arccos(sum_i sqrt(p_i q_i)), here for uniform against (0.7, 0.2, 0.1).
    overlap = np.sum(np.sqrt(np.array([0.7, 0.2, 0.1]) / 3.0))
    assert hits["02: d_FR(uniform, peaked) [rad]"] == pytest.approx(
        2.0 * np.arccos(overlap)
    )


# ----------------------------- Module 02: quantum information ----------- #
def test_s7_bell_quantum_mutual_information_is_two_bits(hits):
    assert hits["02: Bell QMI [bits]"] == pytest.approx(2.0)


def test_s7_bell_conditional_entropy_is_minus_one_bit(hits):
    assert hits["02: Bell S(A|B) [bits]"] == pytest.approx(-1.0)


def test_s7_plus_vs_mixed_quantum_distinguished(hits):
    # Both Umegaki and Bures see what classical KL on diagonals (= 0) misses.
    assert hits["02: S(|+><+| || I/2) [bits]"] == pytest.approx(1.0)
    # Fidelity <+| I/2 |+> = 1/2, so d_B = sqrt(2 - 2 sqrt(1/2)) = sqrt(2 - sqrt(2)).
    assert hits["02: d_B(|+><+|, I/2)"] == pytest.approx(np.sqrt(2.0 - np.sqrt(2.0)))


# ----------------------------- Module 03: Kantorovich ------------------- #
def test_s8_kantorovich_cost_where_no_monge_map_exists(hits):
    # Halves at 0 and 4 feed thirds at 1, 2, 3. The monotone plan moves 1/3 over distance 1
    # and 1/6 over distance 2, from each side: 2 * (1/3 * 1 + 1/6 * 4) = 2.
    assert hits["03: monge-fails LP cost"] == pytest.approx(2.0)


# ----------------------------- Module 03: Wasserstein ------------------- #
def test_s9_closed_form_matches_LP(hits):
    assert hits["03: W2 closed form"] == pytest.approx(hits["03: W2 via LP"], abs=1e-6)
    # Quantile coupling of (0.7, 0.2, 0.1) and (0.1, 0.3, 0.6) on positions 0, 1, 2:
    # 0.3 moves one step, 0.3 two steps and 0.2 one step: W2^2 = 0.3 + 1.2 + 0.2 = 1.7.
    assert hits["03: W2 closed form"] == pytest.approx(np.sqrt(1.7))


def test_s10_small_epsilon_sinkhorn_reaches_the_two_point_optimum(hits):
    # Diagonal states (7/9, 2/9) and (1/4, 3/4) with cost 1 off the diagonal: the optimum
    # moves the excess |7/9 - 1/4| = 19/36 of mass.
    assert hits["03: classical Sinkhorn limit (diag, small eps)"] == pytest.approx(
        19.0 / 36.0, abs=1e-6
    )


# ----------------------------- Modules 03 and 04 (the bridges) ---------- #
def test_s11_bridge_to_quantum_bures(hits):
    # sqrt(Bures-matrix-term) on |+><+| vs I/2 = the Bures distance of 02/13.
    assert hits["03: sqrt(BW matrix) (= d_B from Module 02)"] == pytest.approx(
        hits["02: d_B(|+><+|, I/2)"], abs=1e-6
    )


def test_s13_qot_of_plus_and_mixed_is_one_half(hits):
    # A pure marginal leaves one coupling, the product |+><+| (x) I/2, and the cost
    # operator has diagonal (0, 1, 1, 0): tr(C P) = 1/4 + 1/4.
    assert hits["04: QOT(|+><+|, I/2) > 0"] == pytest.approx(0.5, abs=1e-5)


def test_s13_swap_qot_orthogonal_pures_is_one(hits):
    assert hits["04: SWAP-QOT^2(|0>, |1>) = 1"] == pytest.approx(1.0, abs=1e-3)


def test_s14_amari_quantum_bridge_identity(hits):
    """ε S_Umegaki(P || K) = tr(C P) - ε S(P), and its value at the entropic optimum.

    The equality holds for every state P, since log K = -C/ε; on its own it only checks
    ``umegaki_kl_to_kernel`` against the two terms written out. What can fail on the
    solver is the value: with a pure marginal the optimum is |+><+| (x) I/2, of cost
    1/2 and entropy ln 2, so both sides equal 1/2 - ε ln 2 at ε = 0.4.
    """
    lhs = hits["04: eps * S_Umegaki(P || K)"]
    rhs = hits["04: tr(C P) - eps S(P)"]
    assert lhs == pytest.approx(rhs, abs=1e-9)
    assert rhs == pytest.approx(0.5 - 0.4 * np.log(2.0), abs=1e-6)


# ----------------------------- Smoke: full dictionary present ----------- #
def test_all_expected_hits_present(hits):
    expected_keys = {
        "02: H(fair coin) [bits]",
        "02: I(independent) [bits]",
        "02: I(correlated) [bits]",
        "02: d_FR(uniform, peaked) [rad]",
        "02: mixture midpoint mass in middle",
        "02: W2 midpoint mass in middle",
        "02: Bell QMI [bits]",
        "02: Bell S(A|B) [bits]",
        "02: S(|+><+| || I/2) [bits]",
        "02: d_B(|+><+|, I/2)",
        "03: monge-fails LP cost",
        "03: W2 closed form",
        "03: W2 via LP",
        "03: classical Sinkhorn limit (diag, small eps)",
        "03: sqrt(BW matrix) (= d_B from Module 02)",
        "04: QOT(|+><+|, I/2) > 0",
        "04: SWAP-QOT^2(|0>, |1>) = 1",
        "04: tr(C P) - eps S(P)",
        "04: eps * S_Umegaki(P || K)",
    }
    assert expected_keys <= set(hits.keys())
