"""Gaussian Wasserstein --- the closed form and the bridge to density matrices.

S11 opens the door to M4. For two multivariate Gaussians :math:`\\mathcal{N}(m_0,
\\Sigma_0), \\mathcal{N}(m_1, \\Sigma_1)` on :math:`\\mathbb{R}^d`, the squared
2-Wasserstein distance is closed form (Dowson & Landau, 1982; Olkin & Pukelsheim, 1982;
Givens & Shortt, 1984):

.. math::
    W_2^2 = \\|m_0 - m_1\\|^2 + \\mathrm{tr}(\\Sigma_0) + \\mathrm{tr}(\\Sigma_1)
            - 2\\,\\mathrm{tr}\\!\\sqrt{\\Sigma_0^{1/2}\\,\\Sigma_1\\,\\Sigma_0^{1/2}}.

The matrix term is the **Bures matrix distance** (Bhatia, Jain & Lim, 2019), and when
:math:`\\Sigma_0, \\Sigma_1` are unit-trace PSD matrices it equals the squared Bures
distance between *density matrices* that we already met in S7. **This is the bridge to
quantum OT**: replace covariance matrices by density matrices and the same formula
defines a quantum Wasserstein on states (S13--S14).

For zero-mean Gaussians the optimal transport map is *affine*:
:math:`T(x) = A\\,x` with :math:`A = \\Sigma_0^{-1/2}(\\Sigma_0^{1/2}\\Sigma_1
\\Sigma_0^{1/2})^{1/2}\\Sigma_0^{-1/2}` (the unique SPD matrix that pushes
:math:`\\mathcal{N}(0, \\Sigma_0)` onto :math:`\\mathcal{N}(0, \\Sigma_1)`). The
:math:`W_2` geodesic stays Gaussian: :math:`\\mathcal{N}(m_t, \\Sigma_t)` with
:math:`m_t = (1 - t) m_0 + t m_1` and :math:`\\Sigma_t = M_t \\Sigma_0 M_t^\\top`
where :math:`M_t = (1 - t)\\,I + t\\,A`.

References: D. C. Dowson, B. V. Landau, "The Frechet distance between multivariate
normal distributions", J. Multivar. Anal. 12, 450 (1982); I. Olkin, F. Pukelsheim, "The
distance between two random vectors with given dispersion matrices", Linear Algebra
Appl. 48, 257 (1982); J.-D. Benamou, Y. Brenier, "A computational fluid mechanics
solution to the Monge-Kantorovich mass transfer problem", Numer. Math. 84, 375 (2000);
F. Otto, "The geometry of dissipative evolution equations: the porous medium equation",
Comm. PDE 26, 101 (2001).
"""

from __future__ import annotations

import numpy as np


def _hermitize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    return 0.5 * (M + M.conj().T)


def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
    """Matrix square root of a Hermitian PSD matrix (works for real and complex)."""
    M = _hermitize(M)
    vals, vecs = np.linalg.eigh(M)
    sqrt_vals = np.sqrt(np.clip(vals.real, 0.0, None))
    return (vecs * sqrt_vals) @ vecs.conj().T


def _inv_sqrtm_psd(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Inverse of the matrix square root, thresholded for numerical stability."""
    M = _hermitize(M)
    vals, vecs = np.linalg.eigh(M)
    sqrt_vals = np.sqrt(np.clip(vals.real, 0.0, None))
    inv_sqrt_vals = np.where(sqrt_vals > tol, 1.0 / np.maximum(sqrt_vals, tol), 0.0)
    return (vecs * inv_sqrt_vals) @ vecs.conj().T


def bures_matrix_distance(sigma_0: np.ndarray, sigma_1: np.ndarray) -> float:
    """Bures matrix distance squared ``B^2(Sigma_0, Sigma_1)``.

    Defined as :math:`\\mathrm{tr}(\\Sigma_0) + \\mathrm{tr}(\\Sigma_1) - 2\\,
    \\mathrm{tr}\\sqrt{\\Sigma_0^{1/2}\\Sigma_1\\Sigma_0^{1/2}}` for any pair of
    Hermitian PSD matrices. For **unit-trace** :math:`\\rho_0, \\rho_1` (i.e. density
    matrices) this equals :math:`2(1 - F_U(\\rho_0, \\rho_1)) = d_B^2(\\rho_0, \\rho_1)`
    --- the squared Bures distance we met in S7.
    """
    sigma_0 = _hermitize(np.asarray(sigma_0, dtype=complex))
    sigma_1 = _hermitize(np.asarray(sigma_1, dtype=complex))
    sqrt_s0 = _sqrtm_psd(sigma_0)
    inner = sqrt_s0 @ sigma_1 @ sqrt_s0
    inner_sqrt = _sqrtm_psd(inner)
    value = np.trace(sigma_0) + np.trace(sigma_1) - 2.0 * np.trace(inner_sqrt)
    return float(np.real(value))


def bures_wasserstein_distance(
    mean_0: np.ndarray,
    cov_0: np.ndarray,
    mean_1: np.ndarray,
    cov_1: np.ndarray,
) -> float:
    """Wasserstein-2 distance between two multivariate Gaussians (closed form).

    :math:`W_2^2 = \\|m_0 - m_1\\|^2 + \\text{bures\\_matrix\\_distance}(\\Sigma_0, \\Sigma_1).`
    """
    mean_0 = np.asarray(mean_0, dtype=float).ravel()
    mean_1 = np.asarray(mean_1, dtype=float).ravel()
    mean_term = float(np.sum((mean_0 - mean_1) ** 2))
    matrix_term = bures_matrix_distance(cov_0, cov_1)
    return float(np.sqrt(max(0.0, mean_term + matrix_term)))


def gaussian_ot_map(cov_0: np.ndarray, cov_1: np.ndarray) -> np.ndarray:
    """Affine OT map :math:`A` such that :math:`x \\mapsto A x` pushes
    :math:`\\mathcal{N}(0, \\Sigma_0)` onto :math:`\\mathcal{N}(0, \\Sigma_1)`.

    Closed form: :math:`A = \\Sigma_0^{-1/2}(\\Sigma_0^{1/2}\\Sigma_1\\Sigma_0^{1/2})^{1/2}
    \\Sigma_0^{-1/2}`. The matrix :math:`A` is symmetric positive definite and satisfies
    :math:`A \\Sigma_0 A = \\Sigma_1`.
    """
    cov_0 = np.asarray(cov_0, dtype=float)
    cov_1 = np.asarray(cov_1, dtype=float)
    sqrt_c0 = _sqrtm_psd(cov_0).real
    inv_sqrt_c0 = _inv_sqrtm_psd(cov_0).real
    inner = sqrt_c0 @ cov_1 @ sqrt_c0
    inner_sqrt = _sqrtm_psd(inner).real
    return inv_sqrt_c0 @ inner_sqrt @ inv_sqrt_c0


def gaussian_geodesic(
    mean_0: np.ndarray,
    cov_0: np.ndarray,
    mean_1: np.ndarray,
    cov_1: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wasserstein-2 geodesic in the Gaussian family at time ``t`` in [0, 1].

    Returns ``(mean_t, cov_t)`` of :math:`\\mathcal{N}(m_t, \\Sigma_t)` with
    :math:`m_t = (1 - t) m_0 + t m_1` and :math:`\\Sigma_t = M_t \\Sigma_0 M_t^\\top`,
    where :math:`M_t = (1 - t) I + t A` and :math:`A` is the
    :func:`gaussian_ot_map` between :math:`\\Sigma_0` and :math:`\\Sigma_1`.
    """
    mean_0 = np.asarray(mean_0, dtype=float).ravel()
    mean_1 = np.asarray(mean_1, dtype=float).ravel()
    cov_0 = np.asarray(cov_0, dtype=float)
    cov_1 = np.asarray(cov_1, dtype=float)
    mean_t = (1.0 - t) * mean_0 + t * mean_1
    A = gaussian_ot_map(cov_0, cov_1)
    M_t = (1.0 - t) * np.eye(cov_0.shape[0]) + t * A
    cov_t = M_t @ cov_0 @ M_t.T
    return mean_t, cov_t
