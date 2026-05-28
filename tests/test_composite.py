import numpy as np
import pytest

from qot_course.quantum.composite import (
    apply_channel,
    bell_state,
    depolarizing_channel,
    entanglement_entropy,
    is_cptp,
    partial_trace,
    tensor,
)
from qot_course.quantum.density import density_matrix, maximally_mixed, purity
from qot_course.quantum.states import KET_0, KET_PLUS


def test_tensor_dimension():
    assert tensor(KET_0, KET_0).shape == (4,)


def test_partial_trace_of_product_recovers_factor():
    rho = density_matrix(tensor(KET_0, KET_PLUS))
    reduced_a = partial_trace(rho, keep=[0], dims=[2, 2])
    np.testing.assert_allclose(reduced_a, density_matrix(KET_0), atol=1e-12)


def test_bell_state_parts_are_maximally_mixed():
    rho = density_matrix(bell_state())
    reduced_a = partial_trace(rho, keep=[0], dims=[2, 2])
    np.testing.assert_allclose(reduced_a, maximally_mixed(2), atol=1e-12)


def test_bell_state_has_one_bit_of_entanglement():
    rho = density_matrix(bell_state())
    assert entanglement_entropy(rho, dims=[2, 2]) == pytest.approx(1.0)


def test_product_state_has_no_entanglement():
    rho = density_matrix(tensor(KET_0, KET_PLUS))
    assert entanglement_entropy(rho, dims=[2, 2]) == pytest.approx(0.0, abs=1e-9)


def test_depolarizing_channel_is_cptp_and_shrinks_purity():
    kraus = depolarizing_channel(0.5)
    assert is_cptp(kraus)
    rho = density_matrix(KET_PLUS)
    out = apply_channel(rho, kraus)
    assert purity(out) < purity(rho)
