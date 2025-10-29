"""
Classical Optimal Transport implementation using POT library.

This module provides a unified interface for computing optimal transport
between probability distributions using various algorithms.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Literal
import warnings

import numpy as np
import ot  # Python Optimal Transport library

from src.utils.logger import setup_logger
from src.utils.helpers import validate_array, normalize_distribution

# Setup logger
logger = setup_logger(__name__)


class OptimalTransport:
    """
    Classical Optimal Transport solver.
    
    This class provides methods to compute optimal transport between
    probability distributions using the POT library. It supports both
    entropic regularized (Sinkhorn) and exact optimal transport.
    
    Parameters
    ----------
    method : Literal["sinkhorn", "exact"], optional
        Transport computation method, by default "sinkhorn"
    reg : float, optional
        Entropic regularization parameter (for Sinkhorn), by default 0.01
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Convergence tolerance, by default 1e-9
    verbose : bool, optional
        Enable verbose output, by default False
    
    Attributes
    ----------
    method : str
        Transport computation method
    reg : float
        Entropic regularization parameter
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Verbose output flag
    last_result : Optional[Dict]
        Last computed transport result
    
    Examples
    --------
    >>> ot_solver = OptimalTransport(method="sinkhorn", reg=0.01)
    >>> source = np.random.randn(100, 2)
    >>> target = np.random.randn(150, 2)
    >>> result = ot_solver.compute(source, target)
    >>> print(f"OT cost: {result['cost']:.4f}")
    """
    
    def __init__(
        self,
        method: Literal["sinkhorn", "exact"] = "sinkhorn",
        reg: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-9,
        verbose: bool = False,
    ):
        """Initialize OptimalTransport solver."""
        self.method = method
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.last_result: Optional[Dict] = None
        
        # Validate parameters
        if method not in ["sinkhorn", "exact"]:
            raise ValueError(f"Unknown method: {method}. Use 'sinkhorn' or 'exact'")
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        
        logger.info(
            f"OptimalTransport initialized: method={method}, "
            f"reg={reg}, max_iter={max_iter}"
        )
    
    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
        metric: str = "euclidean",
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        """
        Compute optimal transport between source and target distributions.
        
        Parameters
        ----------
        source : np.ndarray
            Source samples of shape (n_samples_source, n_features)
        target : np.ndarray
            Target samples of shape (n_samples_target, n_features)
        source_weights : Optional[np.ndarray], optional
            Source distribution weights. If None, uniform distribution is used.
        target_weights : Optional[np.ndarray], optional
            Target distribution weights. If None, uniform distribution is used.
        metric : str, optional
            Distance metric for cost matrix, by default "euclidean"
        
        Returns
        -------
        Dict[str, Union[np.ndarray, float, int]]
            Dictionary containing:
            - 'transport_plan': Transport matrix (n_source, n_target)
            - 'cost': Optimal transport cost
            - 'cost_matrix': Cost matrix used for computation
            - 'iterations': Number of iterations (for Sinkhorn)
            - 'source_weights': Source distribution used
            - 'target_weights': Target distribution used
        
        Raises
        ------
        ValueError
            If input arrays have invalid shapes or values
        
        Examples
        --------
        >>> source = np.random.randn(100, 2)
        >>> target = np.random.randn(150, 2)
        >>> result = ot_solver.compute(source, target)
        """
        # Validate inputs
        validate_array(source, name="source", expected_ndim=2)
        validate_array(target, name="target", expected_ndim=2)
        
        if source.shape[1] != target.shape[1]:
            raise ValueError(
                f"Source and target must have same number of features. "
                f"Got source: {source.shape[1]}, target: {target.shape[1]}"
            )
        
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
        
        # Compute cost matrix
        logger.debug(f"Computing {metric} cost matrix")
        cost_matrix = ot.dist(source, target, metric=metric)
        
        # Compute optimal transport
        if self.method == "sinkhorn":
            result = self._compute_sinkhorn(
                source_weights, target_weights, cost_matrix
            )
        else:  # exact
            result = self._compute_exact(
                source_weights, target_weights, cost_matrix
            )
        
        # Add additional information
        result["cost_matrix"] = cost_matrix
        result["source_weights"] = source_weights
        result["target_weights"] = target_weights
        
        # Store result
        self.last_result = result
        
        logger.info(f"OT computation complete: cost={result['cost']:.6f}")
        
        return result
    
    def _compute_sinkhorn(
        self,
        source_weights: np.ndarray,
        target_weights: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        """
        Compute transport using Sinkhorn algorithm.
        
        Parameters
        ----------
        source_weights : np.ndarray
            Source distribution
        target_weights : np.ndarray
            Target distribution
        cost_matrix : np.ndarray
            Cost matrix
        
        Returns
        -------
        Dict
            Transport plan, cost, and number of iterations
        """
        logger.debug("Computing Sinkhorn transport")
        
        # Suppress POT warnings about non-convergence
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Compute transport plan
            transport_plan = ot.sinkhorn(
                source_weights,
                target_weights,
                cost_matrix,
                reg=self.reg,
                numItermax=self.max_iter,
                stopThr=self.tol,
                verbose=self.verbose,
            )
        
        # Compute cost
        cost = np.sum(transport_plan * cost_matrix)
        
        return {
            "transport_plan": transport_plan,
            "cost": float(cost),
            "iterations": self.max_iter,  # POT doesn't return actual iterations
            "method": "sinkhorn",
        }
    
    def _compute_exact(
        self,
        source_weights: np.ndarray,
        target_weights: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        """
        Compute exact optimal transport using linear programming.
        
        Parameters
        ----------
        source_weights : np.ndarray
            Source distribution
        target_weights : np.ndarray
            Target distribution
        cost_matrix : np.ndarray
            Cost matrix
        
        Returns
        -------
        Dict
            Transport plan and cost
        """
        logger.debug("Computing exact transport")
        
        # Compute exact transport
        transport_plan = ot.emd(
            source_weights,
            target_weights,
            cost_matrix,
            numItermax=self.max_iter,
        )
        
        # Compute cost
        cost = np.sum(transport_plan * cost_matrix)
        
        return {
            "transport_plan": transport_plan,
            "cost": float(cost),
            "iterations": 0,  # Exact method doesn't iterate
            "method": "exact",
        }
    
    def wasserstein_distance(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
        metric: str = "euclidean",
        p: int = 2,
    ) -> float:
        """
        Compute p-Wasserstein distance between distributions.
        
        Parameters
        ----------
        source : np.ndarray
            Source samples
        target : np.ndarray
            Target samples
        source_weights : Optional[np.ndarray], optional
            Source weights
        target_weights : Optional[np.ndarray], optional
            Target weights
        metric : str, optional
            Distance metric, by default "euclidean"
        p : int, optional
            Order of Wasserstein distance (1 or 2), by default 2
        
        Returns
        -------
        float
            p-Wasserstein distance
        
        Examples
        --------
        >>> w2_dist = ot_solver.wasserstein_distance(source, target, p=2)
        """
        if p not in [1, 2]:
            raise ValueError(f"p must be 1 or 2, got {p}")
        
        result = self.compute(source, target, source_weights, target_weights, metric)
        
        if p == 1:
            return result["cost"]
        else:  # p == 2
            return np.sqrt(result["cost"])
    
    def barycenter(
        self,
        distributions: list[np.ndarray],
        weights: Optional[np.ndarray] = None,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute Wasserstein barycenter of multiple distributions.
        
        Parameters
        ----------
        distributions : list[np.ndarray]
            List of distributions, each of shape (n_samples, n_features)
        weights : Optional[np.ndarray], optional
            Weights for each distribution. If None, uniform weights are used.
        metric : str, optional
            Distance metric, by default "euclidean"
        
        Returns
        -------
        np.ndarray
            Barycenter distribution
        
        Examples
        --------
        >>> dists = [dist1, dist2, dist3]
        >>> barycenter = ot_solver.barycenter(dists)
        """
        if len(distributions) < 2:
            raise ValueError("Need at least 2 distributions for barycenter")
        
        # Validate all distributions have same shape
        n_samples = distributions[0].shape[0]
        n_features = distributions[0].shape[1]
        
        for i, dist in enumerate(distributions):
            if dist.shape != (n_samples, n_features):
                raise ValueError(
                    f"All distributions must have same shape. "
                    f"Expected {(n_samples, n_features)}, "
                    f"got {dist.shape} at index {i}"
                )
        
        if weights is None:
            weights = np.ones(len(distributions)) / len(distributions)
        else:
            weights = normalize_distribution(weights)
        
        logger.info(f"Computing barycenter of {len(distributions)} distributions")
        
        # Stack distributions into matrix
        distributions_matrix = np.stack(distributions, axis=0)
        
        # Compute barycenter using POT
        # For simplicity, use free support barycenter
        barycenter = ot.lp.free_support_barycenter(
            distributions_matrix,
            weights,
            X_init=None,
            b=None,
            verbose=self.verbose,
        )
        
        return barycenter
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OptimalTransport(method='{self.method}', reg={self.reg}, "
            f"max_iter={self.max_iter}, tol={self.tol})"
        )
