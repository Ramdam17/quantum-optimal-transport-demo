import pytest

from qot_course.quantum_ot.synthesis import course_greatest_hits


@pytest.fixture(scope="module")
def hits():
    return course_greatest_hits()


# ----------------------------- S5 --------------------------------------- #
def test_s5_entropy_of_fair_coin_is_one_bit(hits):
    assert hits["S5: H(fair coin) [bits]"] == pytest.approx(1.0)


def test_s5_kl_is_strictly_positive(hits):
    assert hits["S5: D(p || q) > 0 [bits]"] > 0.0


def test_s5_mutual_information_endpoints(hits):
    assert hits["S5: I(independent) [bits]"] == pytest.approx(0.0, abs=1e-12)
    assert hits["S5: I(correlated) [bits]"] == pytest.approx(1.0)


# ----------------------------- S6 --------------------------------------- #
def test_s6_mixture_midpoint_bimodal_w2_midpoint_unimodal(hits):
    # S6 punchline: mixture-midpoint mass in the middle ≈ 0; W2-midpoint mass ≈ 1.
    assert hits["S6: mixture midpoint mass in middle"] < 0.05
    assert hits["S6: W2 midpoint mass in middle"] > 0.95


def test_s6_fisher_rao_distance_positive(hits):
    assert hits["S6: d_FR(uniform, peaked) [rad]"] > 0.0


# ----------------------------- S7 --------------------------------------- #
def test_s7_bell_quantum_mutual_information_is_two_bits(hits):
    assert hits["S7: Bell QMI [bits]"] == pytest.approx(2.0)


def test_s7_bell_conditional_entropy_is_minus_one_bit(hits):
    assert hits["S7: Bell S(A|B) [bits]"] == pytest.approx(-1.0)


def test_s7_plus_vs_mixed_quantum_distinguished(hits):
    # Both Umegaki and Bures see what classical KL on diagonals (= 0) misses.
    assert hits["S7: S(|+><+| || I/2) [bits]"] == pytest.approx(1.0)
    assert hits["S7: d_B(|+><+|, I/2)"] > 0.0


# ----------------------------- S9 --------------------------------------- #
def test_s9_closed_form_matches_LP(hits):
    assert hits["S9: W2 closed form"] == pytest.approx(
        hits["S9: W2 via LP"], abs=1e-6
    )


# ----------------------------- S11 / S13 / S14 (the bridges) ------------ #
def test_s11_bridge_to_quantum_bures(hits):
    # sqrt(Bures-matrix-term) on |+><+| vs I/2 = the S7 Bures distance.
    assert hits["S11: sqrt(BW matrix) (= d_B from S7)"] == pytest.approx(
        hits["S7: d_B(|+><+|, I/2)"], abs=1e-6
    )


def test_s13_qot_distinguishes_plus_from_mixed(hits):
    assert hits["S13: QOT(|+><+|, I/2) > 0"] > 1e-3


def test_s13_swap_qot_orthogonal_pures_is_one(hits):
    assert hits["S13: SWAP-QOT^2(|0>, |1>) = 1"] == pytest.approx(1.0, abs=1e-3)


def test_s14_amari_quantum_bridge_identity(hits):
    """The most important identity of M4: ε S_Umegaki(P || K) = tr(C P) - ε S(P)."""
    lhs = hits["S14: eps * S_Umegaki(P || K)"]
    rhs = hits["S14: tr(C P) - eps S(P)"]
    assert lhs == pytest.approx(rhs, abs=1e-4)


# ----------------------------- Smoke: full dictionary present ----------- #
def test_all_expected_hits_present(hits):
    expected_keys = {
        "S5: H(fair coin) [bits]",
        "S5: I(independent) [bits]",
        "S5: I(correlated) [bits]",
        "S6: d_FR(uniform, peaked) [rad]",
        "S6: mixture midpoint mass in middle",
        "S6: W2 midpoint mass in middle",
        "S7: Bell QMI [bits]",
        "S7: Bell S(A|B) [bits]",
        "S7: S(|+><+| || I/2) [bits]",
        "S7: d_B(|+><+|, I/2)",
        "S8: monge-fails LP cost",
        "S9: W2 closed form",
        "S9: W2 via LP",
        "S11: sqrt(BW matrix) (= d_B from S7)",
        "S13: QOT(|+><+|, I/2) > 0",
        "S13: SWAP-QOT^2(|0>, |1>) = 1",
        "S14: tr(C P) - eps S(P)",
        "S14: eps * S_Umegaki(P || K)",
    }
    assert expected_keys <= set(hits.keys())
