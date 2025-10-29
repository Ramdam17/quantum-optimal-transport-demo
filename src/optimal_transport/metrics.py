"""
Additional optimal transport metrics and distance computations.

This module provides extended metrics beyond the basic OptimalTransport class,
including sliced Wasserstein distances and various OT-based comparisons.
"""

from typing import Optional, Tuple
import numpy as np
import ot

from src.utils.logger import setup_logger
from src.utils.helpers import validate_array, normalize_distribution

# Setup logger
logger = setup_logger(__name__)


def sliced_wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
    n_projections: int = 50,
    p: int = 2,
    seed: Optional[int] = None,
) -> float:
    """
    Compute sliced Wasserstein distance between distributions.
    
    The sliced Wasserstein distance is computed by projecting distributions
    onto random 1D lines and computing the Wasserstein distance on each
    projection, then averaging.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples of shape (n_samples_source, n_features)
    target : np.ndarray
        Target samples of shape (n_samples_target, n_features)
    n_projections : int, optional
        Number of random projections, by default 50
    p : int, optional
        Order of Wasserstein distance (1 or 2), by default 2
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    float
        Sliced Wasserstein distance
    
    Examples
    --------
    >>> source = np.random.randn(100, 10)
    >>> target = np.random.randn(120, 10)
    >>> sw_dist = sliced_wasserstein_distance(source, target)
    
    References
    ----------
    .. [1] Bonneel et al. (2015). Sliced and Radon Wasserstein Barycenters
       of Measures. Journal of Mathematical Imaging and Vision.
    """
    validate_array(source, name="source", expected_ndim=2)
    validate_array(target, name="target", expected_ndim=2)
    
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"Source and target must have same number of features. "
            f"Got source: {source.shape[1]}, target: {target.shape[1]}"
        )
    
    if p not in [1, 2]:
        raise ValueError(f"p must be 1 or 2, got {p}")
    
    if seed is not None:
        np.random.seed(seed)
    
    n_samples_a = source.shape[0]
    n_samples_b = target.shape[0]
    
    # Uniform weights
    a = np.ones(n_samples_a) / n_samples_a
    b = np.ones(n_samples_b) / n_samples_b
    
    # Use POT's sliced Wasserstein function
    distance = ot.sliced_wasserstein_distance(
        source, target, a, b, n_projections=n_projections, p=p, seed=seed
    )
    
    return float(distance)


def compute_ot_distance_matrix(
    distributions: list[np.ndarray],
    method: str = "wasserstein2",
    **kwargs
) -> np.ndarray:
    """
    Compute pairwise OT distance matrix between multiple distributions.
    
    Parameters
    ----------
    distributions : list[np.ndarray]
        List of distributions, each of shape (n_samples, n_features)
    method : str, optional
        Distance method: 'wasserstein1', 'wasserstein2', or 'sliced',
        by default "wasserstein2"
    **kwargs
        Additional arguments passed to distance function
    
    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (n_distributions, n_distributions)
    
    Examples
    --------
    >>> dists = [dist1, dist2, dist3]
    >>> distance_matrix = compute_ot_distance_matrix(dists)
    """
    from src.optimal_transport.classical import OptimalTransport
    
    n_dists = len(distributions)
    distance_matrix = np.zeros((n_dists, n_dists))
    
    if method in ["wasserstein1", "wasserstein2"]:
        p = 1 if method == "wasserstein1" else 2
        ot_solver = OptimalTransport(method="sinkhorn", **kwargs)
        
        for i in range(n_dists):
            for j in range(i + 1, n_dists):
                dist = ot_solver.wasserstein_distance(
                    distributions[i],
                    distributions[j],
                    p=p
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    elif method == "sliced":
        for i in range(n_dists):
            for j in range(i + 1, n_dists):
                dist = sliced_wasserstein_distance(
                    distributions[i],
                    distributions[j],
                    **kwargs
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'wasserstein1', 'wasserstein2', or 'sliced'"
        )
    
    return distance_matrix


def compute_entropic_regularization_path(
    source: np.ndarray,
    target: np.ndarray,
    reg_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OT cost for different regularization values.
    
    This helps understand the effect of entropic regularization on
    the optimal transport problem.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples
    target : np.ndarray
        Target samples
    reg_values : Optional[np.ndarray], optional
        Regularization values to test. If None, uses logarithmic spacing
        from 0.001 to 1.0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (regularization_values, ot_costs)
    
    Examples
    --------
    >>> reg_vals, costs = compute_entropic_regularization_path(source, target)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(reg_vals, costs)
    >>> plt.xscale('log')
    >>> plt.xlabel('Regularization')
    >>> plt.ylabel('OT Cost')
    
    Notes
    -----
    Very low regularization values (< 0.01) may cause numerical instability
    in the Sinkhorn algorithm. Consider using higher values for stable results.
    """
    from src.optimal_transport.classical import OptimalTransport
    
    if reg_values is None:
        # Use 0.01 to 1.0 range to avoid numerical instability
        reg_values = np.logspace(-2, 0, 20)  # From 0.01 to 1.0
    
    # Warn if any regularization values are very low
    if np.any(reg_values < 0.01):
        logger.warning(
            "Regularization values < 0.01 may cause numerical instability. "
            "Consider using higher values for stable results."
        )
    
    costs = np.zeros(len(reg_values))
    
    for i, reg in enumerate(reg_values):
        ot_solver = OptimalTransport(method="sinkhorn", reg=float(reg))
        result = ot_solver.compute(source, target)
        costs[i] = result["cost"]
    
    logger.info(f"Computed regularization path with {len(reg_values)} values")
    
    return reg_values, costs


def gromov_wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
    target_weights: Optional[np.ndarray] = None,
    loss_fun: str = "square_loss",
    max_iter: int = 1000,
) -> float:
    """
    Compute Gromov-Wasserstein distance between distributions.
    
    Gromov-Wasserstein distance is useful when comparing distributions
    in different metric spaces.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples of shape (n_samples_source, n_features_source)
    target : np.ndarray
        Target samples of shape (n_samples_target, n_features_target)
    source_weights : Optional[np.ndarray], optional
        Source distribution weights
    target_weights : Optional[np.ndarray], optional
        Target distribution weights
    loss_fun : str, optional
        Loss function: 'square_loss' or 'kl_loss', by default "square_loss"
    max_iter : int, optional
        Maximum iterations, by default 1000
    
    Returns
    -------
    float
        Gromov-Wasserstein distance
    
    Examples
    --------
    >>> source = np.random.randn(100, 5)
    >>> target = np.random.randn(120, 8)  # Different feature dims OK!
    >>> gw_dist = gromov_wasserstein_distance(source, target)
    
    References
    ----------
    .. [1] Mémoli, F. (2011). Gromov–Wasserstein distances and the metric
       approach to object matching. Foundations of Computational Mathematics.
    """
    validate_array(source, name="source", expected_ndim=2)
    validate_array(target, name="target", expected_ndim=2)
    
    n_source = source.shape[0]
    n_target = target.shape[0]
    
    # Set up distributions
    if source_weights is None:
        source_weights = np.ones(n_source) / n_source
    else:
        validate_array(source_weights, name="source_weights", expected_ndim=1)
        source_weights = normalize_distribution(source_weights)
    
    if target_weights is None:
        target_weights = np.ones(n_target) / n_target
    else:
        validate_array(target_weights, name="target_weights", expected_ndim=1)
        target_weights = normalize_distribution(target_weights)
    
    # Compute cost matrices (within each space)
    C1 = ot.dist(source, source, metric="euclidean")
    C2 = ot.dist(target, target, metric="euclidean")
    
    # Compute Gromov-Wasserstein distance
    gw_dist, _ = ot.gromov.gromov_wasserstein2(
        C1, C2, source_weights, target_weights,
        loss_fun=loss_fun,
        max_iter=max_iter,
        log=True
    )
    
    return float(gw_dist)


__all__ = [
    "sliced_wasserstein_distance",
    "compute_ot_distance_matrix",
    "compute_entropic_regularization_path",
    "gromov_wasserstein_distance",
]
