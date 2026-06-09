"""Closed-form validation of the public gate + observable API (Plan K)."""

import numpy as np
import pytest

from qot_course.quantum import gates as g
from qot_course.quantum.composite import bell_state
from qot_course.quantum.states import KET_0, KET_1, KET_PLUS, bloch_vector


@pytest.mark.parametrize(
    "U", [g.PAULI_X, g.PAULI_Y, g.PAULI_Z, g.HADAMARD, g.S, g.S_DAG, g.CNOT]
)
def test_all_named_gates_are_unitary(U):
    assert g.is_unitary(U)


def test_rotation_and_phase_gates_unitary():
    for theta in (0.3, 1.0, np.pi):
        assert g.is_unitary(g.rx(theta))
        assert g.is_unitary(g.ry(theta))
        assert g.is_unitary(g.rz(theta))
    assert g.is_unitary(g.phase_gate(0.7))


def test_pauli_x_flips():
    assert np.allclose(g.apply_gate(g.PAULI_X, KET_0), KET_1)


def test_hadamard_makes_plus_and_HZH_is_X():
    assert np.allclose(g.apply_gate(g.HADAMARD, KET_0), KET_PLUS)
    assert np.allclose(g.HADAMARD @ g.PAULI_Z @ g.HADAMARD, g.PAULI_X)


def test_expectation_matches_bloch_vector():
    psi = np.array([0.5, 0.612 + 0.612j])
    psi = psi / np.linalg.norm(psi)
    x, y, z = bloch_vector(psi)
    assert np.isclose(g.expectation(psi, g.PAULI_X), x)
    assert np.isclose(g.expectation(psi, g.PAULI_Y), y)
    assert np.isclose(g.expectation(psi, g.PAULI_Z), z)
    assert np.isclose(g.expectation(KET_PLUS, g.PAULI_X), 1.0)


def test_cnot_builds_bell():
    state00 = np.kron(KET_0, KET_0)
    had_on_control = np.kron(g.HADAMARD, np.eye(2))
    out = g.apply_gate(g.CNOT, g.apply_gate(had_on_control, state00))
    assert np.allclose(out, bell_state())


def test_cnot_truth_table():
    def ket(a, b):
        a_vec = [1, 0] if a == 0 else [0, 1]
        b_vec = [1, 0] if b == 0 else [0, 1]
        return np.kron(a_vec, b_vec).astype(complex)

    assert np.allclose(g.apply_gate(g.CNOT, ket(1, 0)), ket(1, 1))  # control 1 -> flip
    assert np.allclose(g.apply_gate(g.CNOT, ket(0, 1)), ket(0, 1))  # control 0 -> idle
