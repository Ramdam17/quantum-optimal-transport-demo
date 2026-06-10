import numpy as np
from qot_course.quantum_ot.embeddings import (
    amplitude_phase_state,
    covariance_density,
    multifreq_state,
    joint_density_from_states,
)


def _is_state(rho, atol=1e-9):
    return (
        np.allclose(rho, rho.conj().T, atol=atol)
        and abs(np.trace(rho).real - 1.0) < atol
        and np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min() > -atol
    )


def test_multifreq_state_is_normalised_qutrit():
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    psi = multifreq_state(theta, harmonics=(1, 2))
    assert psi.shape == (50, 3)
    np.testing.assert_allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0, atol=1e-12)


def test_joint_density_from_qutrit_states_is_valid():
    rng = np.random.default_rng(0)
    ta = rng.uniform(0, 2 * np.pi, 4000)
    tb = rng.uniform(0, 2 * np.pi, 4000)
    rho = joint_density_from_states(multifreq_state(ta), multifreq_state(tb))
    assert rho.shape == (9, 9) and _is_state(rho)


def test_multifreq_encodes_second_circular_moment():
    # For a CONSTANT phase difference delta, the joint coherence carrying <e^{2i delta}>
    # sits at the (|2>_A|0>_B , |0>_A|2>_B) element. With index = i*3 + j:
    #   |2>_A|0>_B -> 2*3+0 = 6 ; |0>_A|2>_B -> 0*3+2 = 2.
    # rho[6,2] = E[ psi_{20} conj(psi_{02}) ] = (1/9) E[ e^{i(2 theta_A - 2 theta_B)} ]
    #          = (1/9) E[ e^{2i delta} ].
    # (NOTE: rho[8,0] would be the |2,2><0,0| element ∝ e^{2i(theta_A+theta_B)}, which
    #  averages to ZERO for uniform theta_A — do NOT use rho[8,0] for the difference moment.)
    n = 40000
    rng = np.random.default_rng(1)
    ta = rng.uniform(0, 2 * np.pi, n)
    delta = 0.7
    tb = ta - delta
    rho = joint_density_from_states(multifreq_state(ta), multifreq_state(tb))
    coherence_2 = rho[6, 2]
    assert abs(coherence_2) > 0.05  # the 2nd-moment coherence is present
    np.testing.assert_allclose(
        np.angle(coherence_2), 2 * delta, atol=0.05
    )  # and carries 2*delta


def test_first_and_second_moment_live_at_different_elements():
    # First moment <e^{i delta}> lives at (|1>_A|0>_B , |0>_A|1>_B) = rho[3,1].
    n = 40000
    rng = np.random.default_rng(2)
    ta = rng.uniform(0, 2 * np.pi, n)
    delta = 0.7
    tb = ta - delta
    rho = joint_density_from_states(multifreq_state(ta), multifreq_state(tb))
    np.testing.assert_allclose(
        np.angle(rho[3, 1]), delta, atol=0.05
    )  # first moment -> delta
    np.testing.assert_allclose(
        np.angle(rho[6, 2]), 2 * delta, atol=0.05
    )  # second moment -> 2*delta


def test_amplitude_and_covariance_embeddings_are_valid_states():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    psi = amplitude_phase_state(theta, amp=np.abs(np.cos(theta)))
    np.testing.assert_allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0, atol=1e-12)
    assert _is_state(covariance_density(theta))


# --- Research extension (2026-06): weighted multifreq + entangling joint density ---

from qot_course.quantum_ot.embeddings import (  # noqa: E402
    weighted_multifreq_state,
    entangling_joint_density,
    controlled_shift_unitary,
)


def test_weighted_multifreq_equal_weights_matches_multifreq():
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    np.testing.assert_allclose(
        weighted_multifreq_state(theta, harmonics=(1, 2), weights=None),
        multifreq_state(theta, harmonics=(1, 2)),
        atol=1e-12,
    )


def test_weighted_multifreq_is_normalised_per_sample():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    psi = weighted_multifreq_state(theta, harmonics=(1, 2), weights=(1.0, 0.2, 2.0))
    np.testing.assert_allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0, atol=1e-12)


def test_weighted_multifreq_second_moment_channel_closed_form():
    # Constant phase difference delta: rho[6,2] = w0^2 w2^2 <e^{2i delta}> (normalised w).
    # With delta constant, |<e^{2i delta}>| = 1, so |rho[6,2]| = w0^2 w2^2.
    delta = 0.7
    theta_a = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    theta_b = theta_a - delta
    w = np.array([1.0, 0.2, 2.0])
    w = w / np.linalg.norm(w)
    rho = joint_density_from_states(
        weighted_multifreq_state(theta_a, (1, 2), w),
        weighted_multifreq_state(theta_b, (1, 2), w),
    )
    np.testing.assert_allclose(abs(rho[6, 2]), w[0] ** 2 * w[2] ** 2, atol=1e-9)


def test_controlled_shift_is_unitary_and_entangles():
    from qot_course.quantum.composite import entanglement_entropy

    U = controlled_shift_unitary(3)
    assert U.shape == (9, 9)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(9), atol=1e-12)
    # Control in superposition, target in a basis state: |+>_A (x) |0>_B
    # -> (|00> + |11> + |22>)/sqrt(3), a maximally entangled qutrit pair (entropy = log2 3).
    plus = np.ones(3, dtype=complex) / np.sqrt(3)
    ket0 = np.array([1, 0, 0], dtype=complex)
    psi = np.kron(plus, ket0) @ U.T
    rho = np.outer(psi, psi.conj())
    np.testing.assert_allclose(
        entanglement_entropy(rho, dims=[3, 3]), np.log2(3), atol=1e-9
    )


def test_entangling_with_identity_matches_plain_joint():
    rng = np.random.default_rng(1)
    ta = rng.uniform(0, 2 * np.pi, 3000)
    tb = rng.uniform(0, 2 * np.pi, 3000)
    psi_a, psi_b = multifreq_state(ta), multifreq_state(tb)
    plain = joint_density_from_states(psi_a, psi_b)
    ent_id = entangling_joint_density(psi_a, psi_b, np.eye(9, dtype=complex))
    np.testing.assert_allclose(ent_id, plain, atol=1e-12)


def test_entangling_joint_density_is_valid_state():
    rng = np.random.default_rng(2)
    ta = rng.uniform(0, 2 * np.pi, 3000)
    tb = rng.uniform(0, 2 * np.pi, 3000)
    rho = entangling_joint_density(
        multifreq_state(ta), multifreq_state(tb), controlled_shift_unitary(3)
    )
    assert rho.shape == (9, 9) and _is_state(rho)
