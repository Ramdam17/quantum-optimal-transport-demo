"""
Tests for optimal transport visualization module.

This module tests plotting functions for OT visualizations.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from src.visualization.ot_plots import (
    plot_distributions_1d,
    plot_distributions_2d,
    plot_transport_plan,
    plot_cost_matrix,
    plot_transport_arrows_2d,
    plot_ot_comparison,
)


class TestPlotDistributions1D:
    """Test suite for 1D distribution plotting."""
    
    @pytest.fixture
    def distributions_1d(self):
        """Create 1D distributions."""
        np.random.seed(42)
        source = np.random.randn(100)
        target = np.random.randn(150) + 2
        return source, target
    
    def test_plot_1d_basic(self, distributions_1d, tmp_path):
        """Test basic 1D distribution plot."""
        source, target = distributions_1d
        
        fig = plot_distributions_1d(source, target)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)
    
    def test_plot_1d_with_weights(self, distributions_1d):
        """Test 1D plot with custom weights."""
        source, target = distributions_1d
        source_weights = np.ones(len(source)) / len(source)
        target_weights = np.ones(len(target)) / len(target)
        
        fig = plot_distributions_1d(
            source, target,
            source_weights=source_weights,
            target_weights=target_weights
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_1d_with_2d_input(self, distributions_1d):
        """Test 1D plot with 2D input (should flatten)."""
        source, target = distributions_1d
        source_2d = source.reshape(-1, 1)
        target_2d = target.reshape(-1, 1)
        
        fig = plot_distributions_1d(source_2d, target_2d)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_1d_save(self, distributions_1d, tmp_path):
        """Test saving 1D plot to file."""
        source, target = distributions_1d
        save_path = tmp_path / "test_1d.png"
        
        fig = plot_distributions_1d(source, target, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_1d_custom_title(self, distributions_1d):
        """Test 1D plot with custom title."""
        source, target = distributions_1d
        
        fig = plot_distributions_1d(source, target, title="Custom Title")
        
        assert fig.axes[0].get_title() == "Custom Title"
        plt.close(fig)


class TestPlotDistributions2D:
    """Test suite for 2D distribution plotting."""
    
    @pytest.fixture
    def distributions_2d(self):
        """Create 2D distributions."""
        np.random.seed(42)
        source = np.random.randn(100, 2)
        target = np.random.randn(150, 2) + 2
        return source, target
    
    def test_plot_2d_basic(self, distributions_2d):
        """Test basic 2D distribution plot."""
        source, target = distributions_2d
        
        fig = plot_distributions_2d(source, target)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3  # source, target, overlay
        plt.close(fig)
    
    def test_plot_2d_with_weights(self, distributions_2d):
        """Test 2D plot with custom weights."""
        source, target = distributions_2d
        source_weights = np.ones(len(source)) / len(source)
        target_weights = np.ones(len(target)) / len(target)
        
        fig = plot_distributions_2d(
            source, target,
            source_weights=source_weights,
            target_weights=target_weights
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_2d_invalid_dims(self):
        """Test error for non-2D data."""
        source = np.random.randn(100, 3)  # 3D
        target = np.random.randn(150, 3)
        
        with pytest.raises(ValueError, match="must be 2D"):
            plot_distributions_2d(source, target)
    
    def test_plot_2d_save(self, distributions_2d, tmp_path):
        """Test saving 2D plot to file."""
        source, target = distributions_2d
        save_path = tmp_path / "test_2d.png"
        
        fig = plot_distributions_2d(source, target, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotTransportPlan:
    """Test suite for transport plan visualization."""
    
    @pytest.fixture
    def transport_plan(self):
        """Create a sample transport plan."""
        np.random.seed(42)
        plan = np.random.rand(50, 60)
        plan /= plan.sum()  # Normalize
        return plan
    
    def test_plot_transport_plan_basic(self, transport_plan):
        """Test basic transport plan plot."""
        fig = plot_transport_plan(transport_plan)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # main axis + colorbar
        plt.close(fig)
    
    def test_plot_transport_plan_custom_cmap(self, transport_plan):
        """Test transport plan with custom colormap."""
        fig = plot_transport_plan(transport_plan, cmap='plasma')
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_transport_plan_save(self, transport_plan, tmp_path):
        """Test saving transport plan plot."""
        save_path = tmp_path / "transport_plan.png"
        
        fig = plot_transport_plan(transport_plan, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_transport_plan_invalid_shape(self):
        """Test error for invalid shape."""
        plan = np.random.rand(50)  # 1D
        
        with pytest.raises(ValueError):
            plot_transport_plan(plan)


class TestPlotCostMatrix:
    """Test suite for cost matrix visualization."""
    
    @pytest.fixture
    def cost_matrix(self):
        """Create a sample cost matrix."""
        np.random.seed(42)
        return np.random.rand(50, 60)
    
    def test_plot_cost_matrix_basic(self, cost_matrix):
        """Test basic cost matrix plot."""
        fig = plot_cost_matrix(cost_matrix)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # main axis + colorbar
        plt.close(fig)
    
    def test_plot_cost_matrix_custom_cmap(self, cost_matrix):
        """Test cost matrix with custom colormap."""
        fig = plot_cost_matrix(cost_matrix, cmap='coolwarm')
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_cost_matrix_save(self, cost_matrix, tmp_path):
        """Test saving cost matrix plot."""
        save_path = tmp_path / "cost_matrix.png"
        
        fig = plot_cost_matrix(cost_matrix, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotTransportArrows2D:
    """Test suite for 2D transport arrows visualization."""
    
    @pytest.fixture
    def transport_data_2d(self):
        """Create 2D transport data."""
        np.random.seed(42)
        source = np.random.randn(10, 2)
        target = np.random.randn(12, 2) + 2
        transport_plan = np.random.rand(10, 12)
        transport_plan /= transport_plan.sum()
        return source, target, transport_plan
    
    def test_plot_arrows_basic(self, transport_data_2d):
        """Test basic arrow plot."""
        source, target, plan = transport_data_2d
        
        fig = plot_transport_arrows_2d(source, target, plan)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_arrows_custom_threshold(self, transport_data_2d):
        """Test arrow plot with custom threshold."""
        source, target, plan = transport_data_2d
        
        fig = plot_transport_arrows_2d(source, target, plan, threshold=0.05)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_arrows_invalid_dims(self):
        """Test error for non-2D data."""
        source = np.random.randn(10, 3)  # 3D
        target = np.random.randn(12, 3)
        plan = np.random.rand(10, 12)
        
        with pytest.raises(ValueError, match="must be 2D"):
            plot_transport_arrows_2d(source, target, plan)
    
    def test_plot_arrows_save(self, transport_data_2d, tmp_path):
        """Test saving arrow plot."""
        source, target, plan = transport_data_2d
        save_path = tmp_path / "arrows.png"
        
        fig = plot_transport_arrows_2d(source, target, plan, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotOTComparison:
    """Test suite for OT comparison plots."""
    
    @pytest.fixture
    def comparison_results(self):
        """Create sample comparison results."""
        return {
            'Sinkhorn': {'cost': 0.123, 'time': 0.05},
            'Exact': {'cost': 0.120, 'time': 0.15},
            'Sliced': {'cost': 0.125, 'time': 0.02},
        }
    
    def test_plot_comparison_basic(self, comparison_results):
        """Test basic comparison plot."""
        fig = plot_ot_comparison(comparison_results)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_comparison_custom_metric(self, comparison_results):
        """Test comparison with custom metric."""
        fig = plot_ot_comparison(comparison_results, metric='time')
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_comparison_empty_results(self):
        """Test error for empty results."""
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_ot_comparison({})
    
    def test_plot_comparison_save(self, comparison_results, tmp_path):
        """Test saving comparison plot."""
        save_path = tmp_path / "comparison.png"
        
        fig = plot_ot_comparison(comparison_results, save_path=save_path)
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_comparison_missing_metric(self):
        """Test handling of missing metrics (should use NaN)."""
        results = {
            'Method1': {'cost': 0.1},
            'Method2': {},  # Missing cost
        }
        
        fig = plot_ot_comparison(results, metric='cost')
        
        assert isinstance(fig, Figure)
        plt.close(fig)
