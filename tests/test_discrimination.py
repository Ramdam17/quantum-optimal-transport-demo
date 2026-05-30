import pytest
import numpy as np
from qot_course.quantum_ot.capstone import plv, joint_density_matrix, coupling_qmi
from qot_course.quantum_ot.embeddings import multifreq_state, joint_density_from_states
from qot_course.quantum_ot.discrimination import (
    sample_phase_difference,
    matched_plv_ensembles,
)


def test_sampler_hits_target_circular_moments():
    d = sample_phase_difference(80000, a1=0.4, a2=0.3, seed=0)
    np.testing.assert_allclose(np.mean(np.cos(d)), 0.4, atol=0.02)  # first moment
    np.testing.assert_allclose(np.mean(np.cos(2 * d)), 0.3, atol=0.02)  # second moment


def _rich_qmi(theta_a, theta_b):
    """QMI of the multi-frequency (qutrit) joint embedding."""
    return coupling_qmi(
        joint_density_from_states(multifreq_state(theta_a), multifreq_state(theta_b)),
        dims=(3, 3),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_richer_embedding_separates_matched_plv_ensembles(seed):
    """THE THESIS: two ensembles with identical PLV by construction, indistinguishable
    to PLV and to the naive phase embedding, are SEPARATED by the richer (multi-frequency)
    embedding's quantum mutual information --- because they differ in the second circular
    moment a2, which only the richer embedding carries.

    Asserted on absolute bounds (robust to Monte-Carlo noise in the small 'blind' quantities)
    plus an a2-matched control: when a2 is the same in both ensembles, the richer embedding
    does NOT separate them --- proving the separation is caused by a2, not by the embedding
    or by noise.
    """
    # n chosen empirically for seed-robust margins. The rich separation (signal) is a
    # stable ~0.0049 nats at any n; the fragile legs are the *blind* quantities --- naive_gap
    # and especially rich_null, which are single draws of Monte-Carlo noise with a heavy
    # right tail. Sweeping n over 30-50 independent seeds (not just these 5): at n<=8e5 a
    # nontrivial fraction of draws breach rich_null<0.0015 or ratio>5; at n=1_200_000 zero of
    # 50 seeds breach any bound (min rich_gap 0.0043, max rich_null 9.5e-4, min ratio 5.7),
    # and these 5 seeds clear every bound with comfortable margin (min ratio 12.4, max
    # rich_null 3.8e-4). Picking a smaller n that happens to pass seeds 0-4 would just re-hide
    # the seed fragility this test exists to remove; 1.2e6 gives distribution-level headroom.
    n = 1_200_000
    # (a) a2 DIFFERS between the two ensembles
    (a1, b1), (a2, b2) = matched_plv_ensembles(
        n, a1=0.4, a2_low=0.0, a2_high=0.3, seed=seed
    )
    # 1) PLV identical by construction (matched first circular moment)
    assert abs(plv(a1, b1) - plv(a2, b2)) < 0.02
    # 2) the NAIVE phase embedding is blind (its coherence is the first moment only)
    naive_gap = abs(
        coupling_qmi(joint_density_matrix(a1, b1))
        - coupling_qmi(joint_density_matrix(a2, b2))
    )
    assert naive_gap < 0.003
    # 3) the RICHER embedding SEPARATES them (a real signal; analytic value ~= 0.0049 nats)
    rich_gap = abs(_rich_qmi(a1, b1) - _rich_qmi(a2, b2))
    assert rich_gap > 0.003
    # (b) CONTROL: a2 MATCHED in both ensembles -> the richer embedding must NOT separate
    (c1, d1), (c2, d2) = matched_plv_ensembles(
        n, a1=0.4, a2_low=0.3, a2_high=0.3, seed=seed + 100
    )
    rich_null = abs(_rich_qmi(c1, d1) - _rich_qmi(c2, d2))
    assert rich_null < 0.0015
    # 4) the a2-driven separation dwarfs its own matched-a2 null (the clean, robust clincher)
    assert rich_gap > 5.0 * rich_null
