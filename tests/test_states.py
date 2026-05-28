import numpy as np
import pytest

from qot_course.quantum.states import (
    KET_0,
    KET_PLUS,
    bloch_vector,
    born_probabilities,
    qubit_state,
    sample_counts,
)


def test_qubit_state_basis():
    np.testing.assert_allclose(qubit_state(0.0), KET_0, atol=1e-12)


def test_bloch_vector_of_ket0_points_up():
    np.testing.assert_allclose(bloch_vector(KET_0), [0.0, 0.0, 1.0], atol=1e-12)


def test_bloch_vector_of_plus_points_along_x():
    np.testing.assert_allclose(bloch_vector(KET_PLUS), [1.0, 0.0, 0.0], atol=1e-12)


def test_born_probabilities_of_plus_are_uniform():
    probs = born_probabilities(KET_PLUS)
    assert probs["0"] == pytest.approx(0.5)
    assert probs["1"] == pytest.approx(0.5)


def test_sample_counts_sum_to_shots_and_are_reproducible():
    a = sample_counts(KET_PLUS, shots=2048, seed=1)
    b = sample_counts(KET_PLUS, shots=2048, seed=1)
    assert sum(a.values()) == 2048
    assert a == b  # same seed -> same counts
