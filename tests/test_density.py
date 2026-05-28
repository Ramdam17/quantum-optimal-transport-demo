import numpy as np
import pytest

from qot_course.quantum.density import (
    bloch_vector,
    density_from_bloch,
    density_matrix,
    fidelity,
    is_density_matrix,
    maximally_mixed,
    purity,
    trace_distance,
    von_neumann_entropy,
)
from qot_course.quantum.states import KET_0, KET_1, KET_PLUS


def test_pure_state_density_is_valid_and_pure():
    rho = density_matrix(KET_PLUS)
    assert is_density_matrix(rho)
    assert purity(rho) == pytest.approx(1.0)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-9)


def test_maximally_mixed_has_half_purity_and_one_bit_entropy():
    rho = maximally_mixed(2)
    assert purity(rho) == pytest.approx(0.5)
    assert von_neumann_entropy(rho) == pytest.approx(1.0)


def test_plus_and_mixed_share_diagonal_but_differ_offdiagonal():
    plus = density_matrix(KET_PLUS)
    mixed = maximally_mixed(2)
    np.testing.assert_allclose(
        np.diag(plus), np.diag(mixed), atol=1e-12
    )  # same Z-stats
    assert not np.allclose(plus, mixed)  # different states (coherences)


def test_fidelity_and_trace_distance_extremes():
    r0, r1 = density_matrix(KET_0), density_matrix(KET_1)
    assert fidelity(r0, r0) == pytest.approx(1.0, abs=1e-9)
    assert fidelity(r0, r1) == pytest.approx(0.0, abs=1e-9)
    assert trace_distance(r0, r1) == pytest.approx(1.0, abs=1e-9)
    assert trace_distance(r0, r0) == pytest.approx(0.0, abs=1e-9)


def test_bloch_roundtrip():
    rho = density_matrix(KET_PLUS)
    np.testing.assert_allclose(density_from_bloch(bloch_vector(rho)), rho, atol=1e-9)
