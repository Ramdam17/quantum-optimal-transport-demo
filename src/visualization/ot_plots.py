"""
Visualization functions for optimal transport.

This module provides comprehensive plotting utilities for visualizing
optimal transport problems, including distributions, transport plans,
and cost matrices.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from src.utils.logger import setup_logger
from src.utils.helpers import validate_array

logger = setup_logger(__name__)

__all__ = [
    "plot_distributions_1d",
    "plot_distributions_2d",
    "plot_transport_plan",
    "plot_cost_matrix",
    "plot_transport_arrows_2d",
    "plot_ot_comparison",
]


def plot_distributions_1d(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
    target_weights: Optional[np.ndarray] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot 1D source and target distributions.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples of shape (n_samples,) or (n_samples, 1)
    target : np.ndarray
        Target samples of shape (n_samples,) or (n_samples, 1)
    source_weights : Optional[np.ndarray], optional
        Weights for source samples
    target_weights : Optional[np.ndarray], optional
        Weights for target samples
    bins : int, optional
        Number of histogram bins, by default 30
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 4)
    title : Optional[str], optional
        Plot title
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Flatten if 2D
    source = source.flatten()
    target = target.flatten()
    
    validate_array(source, expected_ndim=1)
    validate_array(target, expected_ndim=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    ax.hist(
        source, bins=bins, weights=source_weights,
        alpha=0.6, color='blue', label='Source', density=True
    )
    ax.hist(
        target, bins=bins, weights=target_weights,
        alpha=0.6, color='red', label='Target', density=True
    )
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Source and Target Distributions')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 1D distribution plot to {save_path}")
    
    return fig


def plot_distributions_2d(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
    target_weights: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot 2D source and target distributions as scatter plots.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples of shape (n_samples, 2)
    target : np.ndarray
        Target samples of shape (n_samples, 2)
    source_weights : Optional[np.ndarray], optional
        Weights for source samples (affects point size)
    target_weights : Optional[np.ndarray], optional
        Weights for target samples (affects point size)
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 5)
    title : Optional[str], optional
        Overall plot title
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    validate_array(source, expected_ndim=2)
    validate_array(target, expected_ndim=2)
    
    if source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("Distributions must be 2D for this plot")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Compute point sizes from weights
    source_sizes = (source_weights * 1000) if source_weights is not None else 20
    target_sizes = (target_weights * 1000) if target_weights is not None else 20
    
    # Plot source
    axes[0].scatter(source[:, 0], source[:, 1], s=source_sizes, 
                    alpha=0.6, color='blue', label='Source')
    axes[0].set_title('Source Distribution')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(alpha=0.3)
    
    # Plot target
    axes[1].scatter(target[:, 0], target[:, 1], s=target_sizes,
                    alpha=0.6, color='red', label='Target')
    axes[1].set_title('Target Distribution')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(alpha=0.3)
    
    # Plot both overlaid
    axes[2].scatter(source[:, 0], source[:, 1], s=source_sizes,
                    alpha=0.5, color='blue', label='Source')
    axes[2].scatter(target[:, 0], target[:, 1], s=target_sizes,
                    alpha=0.5, color='red', label='Target')
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 2D distribution plot to {save_path}")
    
    return fig


def plot_transport_plan(
    transport_plan: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot transport plan as a heatmap.
    
    Parameters
    ----------
    transport_plan : np.ndarray
        Transport matrix of shape (n_source, n_target)
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
    title : Optional[str], optional
        Plot title
    cmap : str, optional
        Colormap name, by default 'viridis'
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    validate_array(transport_plan, expected_ndim=2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(transport_plan, cmap=cmap, aspect='auto')
    
    ax.set_xlabel('Target Samples')
    ax.set_ylabel('Source Samples')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transport Mass')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Optimal Transport Plan')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved transport plan plot to {save_path}")
    
    return fig


def plot_cost_matrix(
    cost_matrix: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    cmap: str = 'hot',
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot cost matrix as a heatmap.
    
    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (n_source, n_target)
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
    title : Optional[str], optional
        Plot title
    cmap : str, optional
        Colormap name, by default 'hot'
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    validate_array(cost_matrix, expected_ndim=2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cost_matrix, cmap=cmap, aspect='auto')
    
    ax.set_xlabel('Target Samples')
    ax.set_ylabel('Source Samples')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cost')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Cost Matrix')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cost matrix plot to {save_path}")
    
    return fig


def plot_transport_arrows_2d(
    source: np.ndarray,
    target: np.ndarray,
    transport_plan: np.ndarray,
    threshold: float = 0.01,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot 2D transport with arrows showing mass movement.
    
    Parameters
    ----------
    source : np.ndarray
        Source samples of shape (n_samples, 2)
    target : np.ndarray
        Target samples of shape (n_samples, 2)
    transport_plan : np.ndarray
        Transport matrix of shape (n_source, n_target)
    threshold : float, optional
        Only plot arrows with transport mass > threshold, by default 0.01
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    title : Optional[str], optional
        Plot title
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    validate_array(source, expected_ndim=2)
    validate_array(target, expected_ndim=2)
    validate_array(transport_plan, expected_ndim=2)
    
    if source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("Source and target must be 2D for arrow plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot source and target points
    ax.scatter(source[:, 0], source[:, 1], s=100, alpha=0.6, 
               color='blue', label='Source', zorder=3)
    ax.scatter(target[:, 0], target[:, 1], s=100, alpha=0.6,
               color='red', label='Target', zorder=3)
    
    # Plot arrows for significant transport
    max_mass = transport_plan.max()
    for i in range(source.shape[0]):
        for j in range(target.shape[0]):
            mass = transport_plan[i, j]
            if mass > threshold:
                # Arrow properties scale with transport mass
                alpha = min(0.8, mass / max_mass)
                width = 0.002 * (mass / max_mass)
                
                dx = target[j, 0] - source[i, 0]
                dy = target[j, 1] - source[i, 1]
                
                ax.arrow(
                    source[i, 0], source[i, 1], dx, dy,
                    alpha=alpha, width=width, head_width=0.05,
                    head_length=0.08, fc='green', ec='green',
                    length_includes_head=True, zorder=2
                )
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Transport Plan (threshold={threshold:.3f})')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved transport arrows plot to {save_path}")
    
    return fig


def plot_ot_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'cost',
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Compare multiple OT results (e.g., different methods or regularizations).
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary mapping method names to OT results
        Each result should contain at least the specified metric
    metric : str, optional
        Metric to compare ('cost', 'time', etc.), by default 'cost'
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    title : Optional[str], optional
        Plot title
    save_path : Optional[Union[str, Path]], optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    if not results:
        raise ValueError("Results dictionary cannot be empty")
    
    methods = list(results.keys())
    values = [results[m].get(metric, np.nan) for m in methods]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(methods))
    bars = ax.bar(methods, values, color=colors, alpha=0.7)
    
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Method')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=9
            )
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'OT Comparison: {metric.capitalize()}')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved OT comparison plot to {save_path}")
    
    return fig
