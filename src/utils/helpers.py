"""
Common utility functions and helpers.

This module provides general-purpose utilities used across the project,
including path handling, data validation, and common preprocessing functions.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np


def ensure_path(path: Union[str, Path], create: bool = False) -> Path:
    """
    Convert string to Path and optionally create directory.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path as string or Path object
    create : bool, optional
        Create directory if it doesn't exist, by default False
        
    Returns
    -------
    Path
        Path object
        
    Examples
    --------
    >>> output_dir = ensure_path("outputs/figures", create=True)
    >>> print(output_dir)
    outputs/figures
    """
    path = Path(path)
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def validate_array(
    arr: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_ndim: Optional[int] = None,
    name: str = "array"
) -> None:
    """
    Validate numpy array properties.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    expected_shape : Optional[Tuple[int, ...]], optional
        Expected shape, by default None
    expected_ndim : Optional[int], optional
        Expected number of dimensions, by default None
    name : str, optional
        Name for error messages, by default "array"
        
    Raises
    ------
    TypeError
        If input is not a numpy array
    ValueError
        If array contains NaN/Inf or shape/ndim doesn't match
        
    Examples
    --------
    >>> data = np.random.randn(100, 10)
    >>> validate_array(data, expected_ndim=2, name="input_data")
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
    
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")
    
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected_shape}"
        )
    
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} has {arr.ndim} dimensions, expected {expected_ndim}"
        )


def normalize_distribution(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize array to sum to 1 (convert to probability distribution).
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    axis : Optional[int], optional
        Axis along which to normalize, by default None (whole array)
        
    Returns
    -------
    np.ndarray
        Normalized array
        
    Raises
    ------
    ValueError
        If array contains negative values or sums to zero
        
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4])
    >>> normalized = normalize_distribution(data)
    >>> print(normalized.sum())
    1.0
    """
    if np.any(arr < 0):
        raise ValueError("Array contains negative values, cannot normalize")
    
    total = np.sum(arr, axis=axis, keepdims=True)
    
    if np.any(total == 0):
        raise ValueError("Array sums to zero, cannot normalize")
    
    return arr / total


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seeds for numpy and Python's random module.
    
    Parameters
    ----------
    seed : int
        Random seed value
        
    Examples
    --------
    >>> set_random_seed(42)
    >>> data = np.random.randn(100)  # Reproducible
    """
    np.random.seed(seed)
    import random
    random.seed(seed)


def compute_pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distances between points.
    
    Parameters
    ----------
    X : np.ndarray
        First set of points, shape (n_samples_X, n_features)
    Y : Optional[np.ndarray], optional
        Second set of points, shape (n_samples_Y, n_features)
        If None, computes distances within X, by default None
    metric : str, optional
        Distance metric, by default "euclidean"
        Options: "euclidean", "sqeuclidean", "cosine"
        
    Returns
    -------
    np.ndarray
        Distance matrix, shape (n_samples_X, n_samples_Y)
        
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> Y = np.random.randn(50, 10)
    >>> distances = compute_pairwise_distances(X, Y)
    >>> print(distances.shape)
    (100, 50)
    """
    if Y is None:
        Y = X
    
    if metric == "euclidean":
        # Efficient computation using broadcasting
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = X @ Y.T
        distances = np.sqrt(np.maximum(XX + YY.T - 2*XY, 0))
    
    elif metric == "sqeuclidean":
        # Squared Euclidean distance
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = X @ Y.T
        distances = np.maximum(XX + YY.T - 2*XY, 0)
    
    elif metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
        cosine_sim = X_norm @ Y_norm.T
        distances = 1 - cosine_sim
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Safely divide arrays, replacing division by zero with fill_value.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    fill_value : float, optional
        Value to use when denominator is zero, by default 0.0
        
    Returns
    -------
    np.ndarray
        Result of division
        
    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 0, 4])
    >>> result = safe_divide(a, b, fill_value=0)
    >>> print(result)
    [0.5, 0.0, 0.75]
    """
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def create_batches(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = False
) -> List[np.ndarray]:
    """
    Split data into batches.
    
    Parameters
    ----------
    data : np.ndarray
        Input data, first dimension is batch dimension
    batch_size : int
        Size of each batch
    shuffle : bool, optional
        Shuffle data before batching, by default False
        
    Returns
    -------
    List[np.ndarray]
        List of batches
        
    Examples
    --------
    >>> data = np.arange(100).reshape(100, 1)
    >>> batches = create_batches(data, batch_size=32)
    >>> print(len(batches))
    4
    """
    n_samples = len(data)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        data = data[indices]
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch = data[i:i+batch_size]
        batches.append(batch)
    
    return batches


def format_size(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.
    
    Parameters
    ----------
    size_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Formatted size string
        
    Examples
    --------
    >>> print(format_size(1024))
    1.00 KB
    >>> print(format_size(1048576))
    1.00 MB
    """
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
