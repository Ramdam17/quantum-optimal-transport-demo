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
