import pytest

from qot_course.quantum_ot.synthesis import course_greatest_hits


@pytest.fixture(scope="module")
def hits():
    return course_greatest_hits()


# ----------------------------- Module 02: information ------------------- #
def test_s5_entropy_of_fair_coin_is_one_bit(hits):
    assert hits["02: H(fair coin) [bits]"] == pytest.approx(1.0)


def test_s5_kl_is_strictly_positive(hits):
    assert hits["02: D(p || q) > 0 [bits]"] > 0.0


def test_s5_mutual_information_endpoints(hits):
    assert hits["02: I(independent) [bits]"] == pytest.approx(0.0, abs=1e-12)
    assert hits["02: I(correlated) [bits]"] == pytest.approx(1.0)


# ----------------------------- Module 02: geometry ---------------------- #
def test_s6_mixture_midpoint_bimodal_w2_midpoint_unimodal(hits):
    # The result of 02/09: mixture-midpoint mass in the middle ≈ 0; W2-midpoint mass ≈ 1.
    assert hits["02: mixture midpoint mass in middle"] < 0.05
    assert hits["02: W2 midpoint mass in middle"] > 0.95


def test_s6_fisher_rao_distance_positive(hits):
    assert hits["02: d_FR(uniform, peaked) [rad]"] > 0.0


# ----------------------------- Module 02: quantum information ----------- #
def test_s7_bell_quantum_mutual_information_is_two_bits(hits):
    assert hits["02: Bell QMI [bits]"] == pytest.approx(2.0)


def test_s7_bell_conditional_entropy_is_minus_one_bit(hits):
    assert hits["02: Bell S(A|B) [bits]"] == pytest.approx(-1.0)


def test_s7_plus_vs_mixed_quantum_distinguished(hits):
    # Both Umegaki and Bures see what classical KL on diagonals (= 0) misses.
    assert hits["02: S(|+><+| || I/2) [bits]"] == pytest.approx(1.0)
    assert hits["02: d_B(|+><+|, I/2)"] > 0.0


# ----------------------------- Module 03: Wasserstein ------------------- #
def test_s9_closed_form_matches_LP(hits):
    assert hits["03: W2 closed form"] == pytest.approx(hits["03: W2 via LP"], abs=1e-6)


# ----------------------------- Modules 03 and 04 (the bridges) ---------- #
def test_s11_bridge_to_quantum_bures(hits):
    # sqrt(Bures-matrix-term) on |+><+| vs I/2 = the Bures distance of 02/13.
    assert hits["03: sqrt(BW matrix) (= d_B from Module 02)"] == pytest.approx(
        hits["02: d_B(|+><+|, I/2)"], abs=1e-6
    )


def test_s13_qot_distinguishes_plus_from_mixed(hits):
    assert hits["04: QOT(|+><+|, I/2) > 0"] > 1e-3


def test_s13_swap_qot_orthogonal_pures_is_one(hits):
    assert hits["04: SWAP-QOT^2(|0>, |1>) = 1"] == pytest.approx(1.0, abs=1e-3)


def test_s14_amari_quantum_bridge_identity(hits):
    """The most important identity of module 04: ε S_Umegaki(P || K) = tr(C P) - ε S(P)."""
    lhs = hits["04: eps * S_Umegaki(P || K)"]
    rhs = hits["04: tr(C P) - eps S(P)"]
    assert lhs == pytest.approx(rhs, abs=1e-4)


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
        "03: sqrt(BW matrix) (= d_B from Module 02)",
        "04: QOT(|+><+|, I/2) > 0",
        "04: SWAP-QOT^2(|0>, |1>) = 1",
        "04: tr(C P) - eps S(P)",
        "04: eps * S_Umegaki(P || K)",
    }
    assert expected_keys <= set(hits.keys())
