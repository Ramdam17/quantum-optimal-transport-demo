"""
Tests for classical optimal transport implementation.

This module tests the OptimalTransport class with various scenarios.
"""

import numpy as np
import pytest

from src.optimal_transport.classical import OptimalTransport


class TestOptimalTransport:
    """Test suite for OptimalTransport class."""
    
    @pytest.fixture
    def ot_solver_sinkhorn(self):
        """Create Sinkhorn OT solver."""
        return OptimalTransport(method="sinkhorn", reg=0.01)
    
    @pytest.fixture
    def ot_solver_exact(self):
        """Create exact OT solver."""
        return OptimalTransport(method="exact")
    
    @pytest.fixture
    def simple_1d_distributions(self):
        """Create simple 1D distributions for testing."""
        np.random.seed(42)
        source = np.random.randn(50, 1)
        target = np.random.randn(60, 1)
        return source, target
    
    @pytest.fixture
    def simple_2d_distributions(self):
        """Create simple 2D distributions for testing."""
        np.random.seed(42)
        source = np.random.randn(100, 2)
        target = np.random.randn(120, 2)
        return source, target
    
    def test_initialization_default(self):
        """Test default initialization."""
        ot_solver = OptimalTransport()
        assert ot_solver.method == "sinkhorn"
        assert ot_solver.reg == 0.01
        assert ot_solver.max_iter == 1000
        assert ot_solver.tol == 1e-9
        assert not ot_solver.verbose
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        ot_solver = OptimalTransport(
            method="exact",
            reg=0.05,
            max_iter=500,
            tol=1e-6,
            verbose=True
        )
        assert ot_solver.method == "exact"
        assert ot_solver.reg == 0.05
        assert ot_solver.max_iter == 500
        assert ot_solver.tol == 1e-6
        assert ot_solver.verbose
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            OptimalTransport(method="invalid")
    
    def test_initialization_invalid_reg(self):
        """Test initialization with invalid regularization."""
        with pytest.raises(ValueError, match="reg must be positive"):
            OptimalTransport(reg=-0.01)
    
    def test_compute_sinkhorn_1d(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test Sinkhorn computation on 1D distributions."""
        source, target = simple_1d_distributions
        
        result = ot_solver_sinkhorn.compute(source, target)
        
        # Check result structure
        assert "transport_plan" in result
        assert "cost" in result
        assert "cost_matrix" in result
        assert "source_weights" in result
        assert "target_weights" in result
        
        # Check shapes
        assert result["transport_plan"].shape == (source.shape[0], target.shape[0])
        assert result["cost_matrix"].shape == (source.shape[0], target.shape[0])
        
        # Check properties
        assert result["cost"] >= 0
        assert np.allclose(result["transport_plan"].sum(axis=1), result["source_weights"], atol=1e-6)
        assert np.allclose(result["transport_plan"].sum(axis=0), result["target_weights"], atol=1e-3)
    
    def test_compute_sinkhorn_2d(self, ot_solver_sinkhorn, simple_2d_distributions):
        """Test Sinkhorn computation on 2D distributions."""
        source, target = simple_2d_distributions
        
        result = ot_solver_sinkhorn.compute(source, target)
        
        assert result["cost"] >= 0
        assert result["transport_plan"].shape == (source.shape[0], target.shape[0])
    
    def test_compute_exact_1d(self, ot_solver_exact, simple_1d_distributions):
        """Test exact OT computation on 1D distributions."""
        source, target = simple_1d_distributions
        
        result = ot_solver_exact.compute(source, target)
        
        # Check result structure
        assert "transport_plan" in result
        assert "cost" in result
        assert result["method"] == "exact"
        
        # Check marginal constraints
        assert np.allclose(result["transport_plan"].sum(axis=1), result["source_weights"])
        assert np.allclose(result["transport_plan"].sum(axis=0), result["target_weights"])
    
    def test_compute_with_custom_weights(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test computation with custom distribution weights."""
        source, target = simple_1d_distributions
        
        # Create custom weights
        source_weights = np.random.rand(source.shape[0])
        source_weights /= source_weights.sum()
        target_weights = np.random.rand(target.shape[0])
        target_weights /= target_weights.sum()
        
        result = ot_solver_sinkhorn.compute(
            source, target,
            source_weights=source_weights,
            target_weights=target_weights
        )
        
        assert np.allclose(result["source_weights"], source_weights)
        assert np.allclose(result["target_weights"], target_weights)
    
    def test_compute_invalid_shape(self, ot_solver_sinkhorn):
        """Test computation with mismatched feature dimensions."""
        source = np.random.randn(50, 2)
        target = np.random.randn(60, 3)  # Different feature dimension
        
        with pytest.raises(ValueError, match="same number of features"):
            ot_solver_sinkhorn.compute(source, target)
    
    def test_compute_invalid_weights_shape(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test computation with invalid weight shapes."""
        source, target = simple_1d_distributions
        
        # Wrong number of weights
        bad_weights = np.ones(source.shape[0] + 1)
        
        with pytest.raises((ValueError, IndexError)):
            ot_solver_sinkhorn.compute(
                source, target,
                source_weights=bad_weights
            )
    
    def test_wasserstein_distance_w1(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test W1 Wasserstein distance computation."""
        source, target = simple_1d_distributions
        
        w1_dist = ot_solver_sinkhorn.wasserstein_distance(source, target, p=1)
        
        assert isinstance(w1_dist, (float, np.floating))
        assert w1_dist >= 0
    
    def test_wasserstein_distance_w2(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test W2 Wasserstein distance computation."""
        source, target = simple_1d_distributions
        
        w2_dist = ot_solver_sinkhorn.wasserstein_distance(source, target, p=2)
        
        assert isinstance(w2_dist, (float, np.floating))
        assert w2_dist >= 0
    
    def test_wasserstein_distance_invalid_p(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test Wasserstein distance with invalid p."""
        source, target = simple_1d_distributions
        
        with pytest.raises(ValueError, match="p must be 1 or 2"):
            ot_solver_sinkhorn.wasserstein_distance(source, target, p=3)
    
    def test_wasserstein_identity(self, ot_solver_exact):
        """Test that Wasserstein distance to itself is zero."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        
        w2_dist = ot_solver_exact.wasserstein_distance(data, data, p=2)
        
        assert w2_dist < 1e-6  # Should be very close to zero
    
    def test_convergence_with_regularization(self, simple_1d_distributions):
        """Test that higher regularization gives different results."""
        source, target = simple_1d_distributions
        
        # Use reasonable regularization values to avoid numerical issues
        # Low regularization (but not too low to avoid overflow)
        ot_low_reg = OptimalTransport(method="sinkhorn", reg=0.01)
        result_low = ot_low_reg.compute(source, target)
        
        # High regularization
        ot_high_reg = OptimalTransport(method="sinkhorn", reg=0.1)
        result_high = ot_high_reg.compute(source, target)
        
        # Results should differ due to regularization
        assert not np.allclose(result_low["transport_plan"], result_high["transport_plan"])
        assert result_high["cost"] >= result_low["cost"]  # More regularization => higher cost
    
    def test_reproducibility(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test that results are reproducible."""
        source, target = simple_1d_distributions
        
        result1 = ot_solver_sinkhorn.compute(source, target)
        result2 = ot_solver_sinkhorn.compute(source, target)
        
        assert np.allclose(result1["transport_plan"], result2["transport_plan"])
        assert np.isclose(result1["cost"], result2["cost"])
    
    def test_last_result_stored(self, ot_solver_sinkhorn, simple_1d_distributions):
        """Test that last result is stored."""
        source, target = simple_1d_distributions
        
        assert ot_solver_sinkhorn.last_result is None
        
        result = ot_solver_sinkhorn.compute(source, target)
        
        assert ot_solver_sinkhorn.last_result is not None
        assert np.allclose(ot_solver_sinkhorn.last_result["cost"], result["cost"])
    
    def test_repr(self, ot_solver_sinkhorn):
        """Test string representation."""
        repr_str = repr(ot_solver_sinkhorn)
        assert "OptimalTransport" in repr_str
        assert "sinkhorn" in repr_str
        assert "reg=0.01" in repr_str
    
    def test_different_metrics(self, ot_solver_sinkhorn, simple_2d_distributions):
        """Test computation with different distance metrics."""
        source, target = simple_2d_distributions
        
        # Euclidean
        result_euclidean = ot_solver_sinkhorn.compute(source, target, metric="euclidean")
        
        # Squared Euclidean
        result_sqeuclidean = ot_solver_sinkhorn.compute(source, target, metric="sqeuclidean")
        
        # Results should differ
        assert not np.isclose(result_euclidean["cost"], result_sqeuclidean["cost"])
    
    def test_transport_plan_properties(self, ot_solver_exact, simple_1d_distributions):
        """Test transport plan satisfies mathematical properties."""
        source, target = simple_1d_distributions
        
        result = ot_solver_exact.compute(source, target)
        plan = result["transport_plan"]
        
        # Non-negative
        assert np.all(plan >= 0)
        
        # Marginal constraints
        assert np.allclose(plan.sum(axis=1), result["source_weights"], atol=1e-6)
        assert np.allclose(plan.sum(axis=0), result["target_weights"], atol=1e-6)
        
        # Conservation of mass
        assert np.isclose(plan.sum(), 1.0, atol=1e-6)
