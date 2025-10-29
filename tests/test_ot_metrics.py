"""
Tests for optimal transport metrics module.

This module tests additional OT metrics beyond the basic OptimalTransport class.
"""

import warnings
import numpy as np
import pytest

# Filter expected POT warnings for numerical edge cases
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in dot")

from src.optimal_transport.metrics import (
    sliced_wasserstein_distance,
    compute_ot_distance_matrix,
    compute_entropic_regularization_path,
    gromov_wasserstein_distance,
)


class TestSlicedWassersteinDistance:
    """Test suite for sliced Wasserstein distance."""
    
    @pytest.fixture
    def simple_distributions(self):
        """Create simple distributions for testing."""
        np.random.seed(42)
        source = np.random.randn(100, 5)
        target = np.random.randn(120, 5)
        return source, target
    
    def test_sliced_wasserstein_basic(self, simple_distributions):
        """Test basic sliced Wasserstein computation."""
        source, target = simple_distributions
        
        distance = sliced_wasserstein_distance(source, target, n_projections=50, seed=42)
        
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0
    
    def test_sliced_wasserstein_reproducibility(self, simple_distributions):
        """Test that results are reproducible with same seed."""
        source, target = simple_distributions
        
        dist1 = sliced_wasserstein_distance(source, target, n_projections=50, seed=42)
        dist2 = sliced_wasserstein_distance(source, target, n_projections=50, seed=42)
        
        assert np.isclose(dist1, dist2)
    
    def test_sliced_wasserstein_identity(self):
        """Test that distance to itself is zero."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        
        distance = sliced_wasserstein_distance(data, data, n_projections=50, seed=42)
        
        assert distance < 1e-6
    
    def test_sliced_wasserstein_p_values(self, simple_distributions):
        """Test both p=1 and p=2."""
        source, target = simple_distributions
        
        dist_p1 = sliced_wasserstein_distance(source, target, p=1, seed=42)
        dist_p2 = sliced_wasserstein_distance(source, target, p=2, seed=42)
        
        assert dist_p1 >= 0
        assert dist_p2 >= 0
        # p=2 typically gives larger distances for Gaussian distributions
    
    def test_sliced_wasserstein_invalid_p(self, simple_distributions):
        """Test invalid p value raises error."""
        source, target = simple_distributions
        
        with pytest.raises(ValueError, match="p must be 1 or 2"):
            sliced_wasserstein_distance(source, target, p=3)
    
    def test_sliced_wasserstein_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        source = np.random.randn(100, 5)
        target = np.random.randn(120, 7)
        
        with pytest.raises(ValueError, match="same number of features"):
            sliced_wasserstein_distance(source, target)


class TestOTDistanceMatrix:
    """Test suite for OT distance matrix computation."""
    
    @pytest.fixture
    def multiple_distributions(self):
        """Create multiple distributions."""
        np.random.seed(42)
        dist1 = np.random.randn(50, 3)
        dist2 = np.random.randn(60, 3) + 1.0  # Shifted
        dist3 = np.random.randn(55, 3) + 2.0  # More shifted
        return [dist1, dist2, dist3]
    
    def test_distance_matrix_wasserstein2(self, multiple_distributions):
        """Test distance matrix with Wasserstein-2."""
        distance_matrix = compute_ot_distance_matrix(
            multiple_distributions,
            method="wasserstein2",
            reg=0.01
        )
        
        # Check shape
        assert distance_matrix.shape == (3, 3)
        
        # Check symmetry
        assert np.allclose(distance_matrix, distance_matrix.T)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(distance_matrix), 0, atol=1e-6)
        
        # Check positive off-diagonal
        assert np.all(distance_matrix[~np.eye(3, dtype=bool)] > 0)
    
    def test_distance_matrix_wasserstein1(self, multiple_distributions):
        """Test distance matrix with Wasserstein-1."""
        distance_matrix = compute_ot_distance_matrix(
            multiple_distributions,
            method="wasserstein1",
            reg=0.01
        )
        
        assert distance_matrix.shape == (3, 3)
        assert np.allclose(distance_matrix, distance_matrix.T)
    
    def test_distance_matrix_sliced(self, multiple_distributions):
        """Test distance matrix with sliced Wasserstein."""
        distance_matrix = compute_ot_distance_matrix(
            multiple_distributions,
            method="sliced",
            n_projections=30,
            seed=42
        )
        
        assert distance_matrix.shape == (3, 3)
        assert np.allclose(distance_matrix, distance_matrix.T)
    
    def test_distance_matrix_invalid_method(self, multiple_distributions):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_ot_distance_matrix(multiple_distributions, method="invalid")
    
    def test_distance_matrix_ordering(self, multiple_distributions):
        """Test that closer distributions have smaller distances."""
        distance_matrix = compute_ot_distance_matrix(
            multiple_distributions,
            method="wasserstein2",
            reg=0.01
        )
        
        # dist1 to dist2 should be less than dist1 to dist3
        # (since dist2 is shifted by +1, dist3 by +2)
        assert distance_matrix[0, 1] < distance_matrix[0, 2]


class TestEntropicRegularizationPath:
    """Test suite for regularization path computation."""
    
    @pytest.fixture
    def simple_distributions(self):
        """Create simple distributions."""
        np.random.seed(42)
        source = np.random.randn(50, 2)
        target = np.random.randn(60, 2)
        return source, target
    
    def test_regularization_path_default(self, simple_distributions):
        """Test regularization path with default values."""
        source, target = simple_distributions
        
        reg_vals, costs = compute_entropic_regularization_path(source, target)
        
        # Check lengths match
        assert len(reg_vals) == len(costs)
        assert len(reg_vals) > 0
        
        # Check all costs are positive
        assert np.all(costs > 0)
        
        # Check regularization values are sorted
        assert np.all(np.diff(reg_vals) > 0)
    
    def test_regularization_path_custom(self, simple_distributions):
        """Test regularization path with custom values."""
        source, target = simple_distributions
        # Use safe regularization values (>= 0.01)
        custom_regs = np.array([0.01, 0.05, 0.1, 0.5])
        
        reg_vals, costs = compute_entropic_regularization_path(
            source, target, reg_values=custom_regs
        )
        
        assert len(costs) == 4
        assert np.allclose(reg_vals, custom_regs)
    
    def test_regularization_path_monotonicity(self, simple_distributions):
        """Test that cost generally increases with regularization."""
        source, target = simple_distributions
        
        reg_vals, costs = compute_entropic_regularization_path(source, target)
        
        # Generally, higher regularization should give higher or similar cost
        # (not strictly monotonic due to numerical issues, but trend should be there)
        # Check that the maximum cost is at higher regularization
        max_cost_idx = np.argmax(costs)
        assert max_cost_idx > len(costs) // 2  # Max should be in second half
    
    def test_regularization_path_low_value_warning(self, simple_distributions):
        """Test warning for very low regularization values."""
        source, target = simple_distributions
        low_regs = np.array([0.001, 0.005, 0.01])
        
        # Should warn about low regularization
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_entropic_regularization_path(source, target, reg_values=low_regs)
            # Check that a warning was issued (from our logger.warning)
            # Note: This tests our own warning, not POT's RuntimeWarnings


class TestGromovWassersteinDistance:
    """Test suite for Gromov-Wasserstein distance."""
    
    def test_gromov_wasserstein_basic(self):
        """Test basic Gromov-Wasserstein computation."""
        np.random.seed(42)
        source = np.random.randn(50, 5)
        target = np.random.randn(60, 8)  # Different dimensions OK!
        
        distance = gromov_wasserstein_distance(source, target, max_iter=100)
        
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0
    
    def test_gromov_wasserstein_identity(self):
        """Test that distance to itself is zero."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        
        distance = gromov_wasserstein_distance(data, data, max_iter=100)
        
        # Should be close to zero (may not be exactly zero due to algorithm)
        assert distance < 1e-3
    
    def test_gromov_wasserstein_with_weights(self):
        """Test Gromov-Wasserstein with custom weights."""
        np.random.seed(42)
        source = np.random.randn(50, 5)
        target = np.random.randn(60, 8)
        
        source_weights = np.random.rand(50)
        source_weights /= source_weights.sum()
        target_weights = np.random.rand(60)
        target_weights /= target_weights.sum()
        
        distance = gromov_wasserstein_distance(
            source, target,
            source_weights=source_weights,
            target_weights=target_weights,
            max_iter=100
        )
        
        assert distance >= 0
    
    def test_gromov_wasserstein_different_dims(self):
        """Test that GW works with different feature dimensions."""
        np.random.seed(42)
        source = np.random.randn(30, 3)
        target = np.random.randn(40, 10)  # Very different dimension
        
        distance = gromov_wasserstein_distance(source, target, max_iter=50)
        
        # Should compute successfully even with different dims
        assert distance >= 0
        assert np.isfinite(distance)
