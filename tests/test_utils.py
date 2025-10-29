"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import logging

from src.utils.helpers import (
    ensure_path,
    validate_array,
    normalize_distribution,
    set_random_seed,
    compute_pairwise_distances,
    safe_divide,
    create_batches,
    format_size
)


class TestEnsurePath:
    """Tests for ensure_path function."""
    
    def test_string_to_path(self):
        """Test converting string to Path."""
        result = ensure_path("test/path")
        assert isinstance(result, Path)
        assert str(result) == "test/path"
    
    def test_create_directory(self):
        """Test creating directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            result = ensure_path(new_dir, create=True)
            assert result.exists()
            assert result.is_dir()


class TestValidateArray:
    """Tests for validate_array function."""
    
    def test_valid_array(self):
        """Test validation of valid array."""
        arr = np.random.randn(10, 5)
        # Should not raise
        validate_array(arr, expected_ndim=2)
    
    def test_non_array_raises(self):
        """Test error for non-array input."""
        with pytest.raises(TypeError, match="must be a numpy array"):
            validate_array([1, 2, 3])
    
    def test_nan_raises(self):
        """Test error for NaN values."""
        arr = np.array([1, 2, np.nan, 4])
        with pytest.raises(ValueError, match="NaN"):
            validate_array(arr)
    
    def test_inf_raises(self):
        """Test error for Inf values."""
        arr = np.array([1, 2, np.inf, 4])
        with pytest.raises(ValueError, match="Inf"):
            validate_array(arr)
    
    def test_wrong_shape_raises(self):
        """Test error for wrong shape."""
        arr = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="shape"):
            validate_array(arr, expected_shape=(10, 3))
    
    def test_wrong_ndim_raises(self):
        """Test error for wrong number of dimensions."""
        arr = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="dimensions"):
            validate_array(arr, expected_ndim=3)


class TestNormalizeDistribution:
    """Tests for normalize_distribution function."""
    
    def test_normalize_1d(self):
        """Test normalizing 1D array."""
        arr = np.array([1, 2, 3, 4])
        result = normalize_distribution(arr)
        assert np.isclose(result.sum(), 1.0)
        assert result.shape == arr.shape
    
    def test_normalize_2d_axis_none(self):
        """Test normalizing entire 2D array."""
        arr = np.random.rand(5, 4)
        result = normalize_distribution(arr)
        assert np.isclose(result.sum(), 1.0)
    
    def test_normalize_2d_axis_1(self):
        """Test normalizing 2D array along axis 1."""
        arr = np.random.rand(5, 4)
        result = normalize_distribution(arr, axis=1)
        assert np.allclose(result.sum(axis=1), 1.0)
    
    def test_negative_raises(self):
        """Test error for negative values."""
        arr = np.array([1, -2, 3])
        with pytest.raises(ValueError, match="negative"):
            normalize_distribution(arr)
    
    def test_zero_sum_raises(self):
        """Test error for zero sum."""
        arr = np.zeros(5)
        with pytest.raises(ValueError, match="sums to zero"):
            normalize_distribution(arr)


class TestSetRandomSeed:
    """Tests for set_random_seed function."""
    
    def test_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_random_seed(42)
        result1 = np.random.randn(10)
        
        set_random_seed(42)
        result2 = np.random.randn(10)
        
        np.testing.assert_array_equal(result1, result2)


class TestComputePairwiseDistances:
    """Tests for compute_pairwise_distances function."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        distances = compute_pairwise_distances(X, metric="euclidean")
        
        # Distance from [0,0] to [1,1] should be sqrt(2)
        assert np.isclose(distances[0, 1], np.sqrt(2))
        
        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)
    
    def test_squared_euclidean(self):
        """Test squared Euclidean distance."""
        X = np.array([[0, 0], [1, 1]])
        distances = compute_pairwise_distances(X, metric="sqeuclidean")
        
        # Squared distance from [0,0] to [1,1] should be 2
        assert np.isclose(distances[0, 1], 2.0)
    
    def test_cosine_distance(self):
        """Test cosine distance computation."""
        X = np.array([[1, 0], [0, 1], [1, 1]])
        distances = compute_pairwise_distances(X, metric="cosine")
        
        # Cosine distance between orthogonal vectors should be 1
        assert np.isclose(distances[0, 1], 1.0)
    
    def test_two_arrays(self):
        """Test distance between two different arrays."""
        X = np.random.randn(10, 5)
        Y = np.random.randn(15, 5)
        distances = compute_pairwise_distances(X, Y)
        
        assert distances.shape == (10, 15)
    
    def test_invalid_metric_raises(self):
        """Test error for invalid metric."""
        X = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_pairwise_distances(X, metric="invalid")


class TestSafeDivide:
    """Tests for safe_divide function."""
    
    def test_normal_division(self):
        """Test normal division."""
        a = np.array([4, 6, 8])
        b = np.array([2, 3, 4])
        result = safe_divide(a, b)
        
        np.testing.assert_array_equal(result, [2, 2, 2])
    
    def test_division_by_zero(self):
        """Test division by zero with fill value."""
        a = np.array([1, 2, 3])
        b = np.array([2, 0, 4])
        result = safe_divide(a, b, fill_value=0)
        
        assert result[0] == 0.5
        assert result[1] == 0.0  # Fill value
        assert result[2] == 0.75
    
    def test_custom_fill_value(self):
        """Test custom fill value."""
        a = np.array([1, 2, 3])
        b = np.array([0, 0, 0])
        result = safe_divide(a, b, fill_value=-1)
        
        np.testing.assert_array_equal(result, [-1, -1, -1])


class TestCreateBatches:
    """Tests for create_batches function."""
    
    def test_exact_batches(self):
        """Test when data divides evenly into batches."""
        data = np.arange(100).reshape(100, 1)
        batches = create_batches(data, batch_size=25)
        
        assert len(batches) == 4
        assert all(len(batch) == 25 for batch in batches)
    
    def test_uneven_batches(self):
        """Test when data doesn't divide evenly."""
        data = np.arange(100).reshape(100, 1)
        batches = create_batches(data, batch_size=30)
        
        assert len(batches) == 4
        assert len(batches[0]) == 30
        assert len(batches[-1]) == 10  # Last batch smaller
    
    def test_shuffle(self):
        """Test shuffling before batching."""
        data = np.arange(100).reshape(100, 1)
        
        # Set seed for reproducibility
        np.random.seed(42)
        batches1 = create_batches(data.copy(), batch_size=25, shuffle=True)
        
        np.random.seed(42)
        batches2 = create_batches(data.copy(), batch_size=25, shuffle=True)
        
        # Same seed should produce same batches
        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1, b2)


class TestFormatSize:
    """Tests for format_size function."""
    
    def test_bytes(self):
        """Test formatting bytes."""
        assert format_size(512) == "512.00 B"
    
    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.00 KB"
        assert format_size(2048) == "2.00 KB"
    
    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1048576) == "1.00 MB"
    
    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1073741824) == "1.00 GB"
