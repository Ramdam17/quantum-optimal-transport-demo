"""Classical information theory: entropy, KL, mutual & transfer information."""

from __future__ import annotations

import numpy as np


def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy H(p) = -sum p log p (in bits by default)."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Kullback-Leibler divergence D(p || q) = sum p log(p/q)."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    mask = p > 0
    if np.any(q[mask] == 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])) / np.log(base))


def mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """Mutual information I(X;Y) from a 2-D joint distribution array."""
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    return float(
        shannon_entropy(px, base)
        + shannon_entropy(py, base)
        - shannon_entropy(joint, base)
    )


def conditional_mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """Conditional mutual information I(X;Y|Z) from a 3-D joint array (axes X, Y, Z)."""
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    h_xz = shannon_entropy(joint.sum(axis=1), base)  # marginalise Y
    h_yz = shannon_entropy(joint.sum(axis=0), base)  # marginalise X
    h_z = shannon_entropy(joint.sum(axis=(0, 1)), base)
    h_xyz = shannon_entropy(joint, base)
    return float(h_xz + h_yz - h_z - h_xyz)


def _joint_counts_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Empirical joint distribution of three integer-symbol sequences."""
    n_sym = int(max(x.max(), y.max(), z.max())) + 1
    counts = np.zeros((n_sym, n_sym, n_sym), dtype=float)
    np.add.at(counts, (x, y, z), 1.0)
    return counts


def transfer_entropy(
    source: np.ndarray, target: np.ndarray, base: float = 2.0
) -> float:
    """Transfer entropy TE_{source -> target} = I(target_{t+1}; source_t | target_t).

    For integer-symbol sequences with lag 1 (Schreiber, 2000).
    """
    source = np.asarray(source).astype(int)
    target = np.asarray(target).astype(int)
    target_next = target[1:]
    source_past = source[:-1]
    target_past = target[:-1]
    # axes: X = target_next, Y = source_past, Z = target_past
    joint = _joint_counts_3d(target_next, source_past, target_past)
    return conditional_mutual_information(joint, base)
