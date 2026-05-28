import numpy as np
import pytest

from qot_course.infotheory.classical import (
    conditional_mutual_information,
    kl_divergence,
    mutual_information,
    shannon_entropy,
    transfer_entropy,
)


def test_entropy_of_fair_coin_is_one_bit():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_kl_is_zero_iff_equal_and_nonnegative():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.6, 0.3])
    assert kl_divergence(p, p) == pytest.approx(0.0)
    assert kl_divergence(p, q) > 0.0


def test_mutual_information_independent_is_zero_correlated_is_one():
    independent = np.outer([0.5, 0.5], [0.5, 0.5])
    assert mutual_information(independent) == pytest.approx(0.0, abs=1e-12)
    correlated = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert mutual_information(correlated) == pytest.approx(1.0)


def test_conditional_mutual_information_nonnegative():
    rng = np.random.default_rng(3)
    joint = rng.random((2, 2, 2))
    joint /= joint.sum()
    assert conditional_mutual_information(joint) >= -1e-12


def test_transfer_entropy_is_directional():
    rng = np.random.default_rng(0)
    source = rng.integers(0, 2, size=4000)
    target = np.empty_like(source)
    target[0] = 0
    target[1:] = source[:-1]  # target copies source with lag 1
    te_fwd = transfer_entropy(source, target)  # source -> target
    te_bwd = transfer_entropy(target, source)  # target -> source
    assert te_fwd > 0.8
    assert te_bwd < 0.1
    assert te_fwd > te_bwd
