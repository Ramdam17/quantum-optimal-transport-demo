"""
Genetics simulator for gene expression data.

This module simulates gene expression profiles for population comparison
studies, with realistic distributions and differential expression patterns.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy import stats

from src.data.base_simulator import BaseSimulator
from src.utils.helpers import validate_array


class GeneticsSimulator(BaseSimulator):
    """
    Simulate gene expression data for population comparison.
    
    This simulator generates gene expression profiles with controllable
    differential expression patterns between populations.
    
    Parameters
    ----------
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    **kwargs
        Additional simulator parameters
        
    Examples
    --------
    >>> simulator = GeneticsSimulator(seed=42)
    >>> data = simulator.generate(n_genes=100, n_samples_per_pop=200)
    >>> print(data['population_A_expression'].shape)
    (200, 100)
    """
    
    def generate(
        self,
        n_genes: int = 100,
        n_samples_per_pop: int = 200,
        n_populations: int = 2,
        population_names: Optional[List[str]] = None,
        mean_expression: float = 5.0,
        variance: float = 2.0,
        n_differential_genes: int = 20,
        differential_magnitude: float = 2.0,
        differential_genes_indices: Optional[List[int]] = None,
        technical_noise: float = 0.1,
        biological_noise: float = 0.2,
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate simulated gene expression data.
        
        Parameters
        ----------
        n_genes : int, optional
            Number of genes, by default 100
        n_samples_per_pop : int, optional
            Samples per population, by default 200
        n_populations : int, optional
            Number of populations, by default 2
        population_names : Optional[List[str]], optional
            Population names, by default None (['population_A', 'population_B'])
        mean_expression : float, optional
            Mean expression level (log2 scale), by default 5.0
        variance : float, optional
            Expression variance, by default 2.0
        n_differential_genes : int, optional
            Number of differentially expressed genes, by default 20
        differential_magnitude : float, optional
            Fold change magnitude (log2 scale), by default 2.0
        differential_genes_indices : Optional[List[int]], optional
            Indices of differential genes, by default None (evenly spaced)
        technical_noise : float, optional
            Technical variation std dev, by default 0.1
        biological_noise : float, optional
            Biological variation std dev, by default 0.2
        normalize : bool, optional
            Apply quantile normalization, by default True
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - '{population}_expression': Expression arrays for each population
            - 'gene_names': List of gene identifiers
            - 'differential_genes': Indices of differential genes
            - 'fold_changes': Fold changes for differential genes
            - Parameters used for generation
        """
        self.logger.info(
            f"Generating gene expression data: {n_populations} populations, "
            f"{n_genes} genes, {n_samples_per_pop} samples per population"
        )
        
        # Default population names
        if population_names is None:
            population_names = [f"population_{chr(65+i)}" for i in range(n_populations)]
        
        if len(population_names) != n_populations:
            raise ValueError(
                f"Need {n_populations} population names, got {len(population_names)}"
            )
        
        # Generate gene names
        gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
        
        # Determine differential genes
        if differential_genes_indices is None:
            # Evenly spaced differential genes
            differential_genes_indices = list(
                range(0, n_genes, max(1, n_genes // n_differential_genes))
            )[:n_differential_genes]
        
        differential_genes_indices = differential_genes_indices[:n_differential_genes]
        
        # Generate base expression profile (log-normal distribution)
        base_expression = np.random.normal(
            mean_expression,
            variance,
            (n_samples_per_pop, n_genes)
        )
        
        # Generate expression for each population
        population_data = {}
        fold_changes = np.zeros(n_genes)
        
        for pop_idx, pop_name in enumerate(population_names):
            # Start with base expression
            expression = base_expression.copy()
            
            # Add biological noise (between-sample variation)
            expression += np.random.normal(
                0,
                biological_noise,
                (n_samples_per_pop, n_genes)
            )
            
            # Add differential expression for population 2+
            if pop_idx > 0:
                for gene_idx in differential_genes_indices:
                    # Random up or down regulation
                    direction = np.random.choice([-1, 1])
                    fold_change = direction * differential_magnitude
                    fold_changes[gene_idx] = fold_change
                    
                    # Apply fold change
                    expression[:, gene_idx] += fold_change
            
            # Add technical noise
            expression += np.random.normal(
                0,
                technical_noise,
                (n_samples_per_pop, n_genes)
            )
            
            # Convert from log space to linear (simulate RNA-seq counts)
            expression = 2 ** expression
            
            # Ensure non-negative
            expression = np.maximum(expression, 0.1)
            
            # Quantile normalization (common preprocessing)
            if normalize:
                expression = self._quantile_normalize(expression)
            
            population_data[f"{pop_name}_expression"] = expression
        
        # Store data
        self.data = {
            **population_data,
            'gene_names': gene_names,
            'differential_genes': differential_genes_indices,
            'fold_changes': fold_changes,
            'n_genes': n_genes,
            'n_samples_per_pop': n_samples_per_pop,
            'n_populations': n_populations,
            'population_names': population_names,
            'mean_expression': mean_expression,
            'variance': variance,
            'n_differential_genes': n_differential_genes,
            'differential_magnitude': differential_magnitude,
            'technical_noise': technical_noise,
            'biological_noise': biological_noise,
            'normalized': normalize
        }
        
        # Validate
        self.validate(self.data)
        
        self.logger.info("Gene expression data generated successfully")
        return self.data
    
    def _quantile_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply quantile normalization to expression data.
        
        Parameters
        ----------
        data : np.ndarray
            Expression matrix (samples x genes)
            
        Returns
        -------
        np.ndarray
            Normalized expression matrix
        """
        # Get ranks
        rank_data = np.argsort(np.argsort(data, axis=0), axis=0)
        
        # Compute mean quantile
        sorted_data = np.sort(data, axis=0)
        mean_quantile = np.mean(sorted_data, axis=1)
        
        # Assign mean quantile values based on ranks
        normalized = mean_quantile[rank_data]
        
        return normalized
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated gene expression data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate
            
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        population_names = data.get('population_names', [])
        n_samples = data.get('n_samples_per_pop')
        n_genes = data.get('n_genes')
        
        # Check each population's expression data
        for pop_name in population_names:
            key = f"{pop_name}_expression"
            if key not in data:
                raise ValueError(f"Missing expression data for: {pop_name}")
            
            expression = data[key]
            validate_array(expression, expected_ndim=2, name=key)
            
            if expression.shape != (n_samples, n_genes):
                raise ValueError(
                    f"{key} has wrong shape: {expression.shape}, "
                    f"expected ({n_samples}, {n_genes})"
                )
            
            # Check for non-negative values (expression can't be negative)
            if np.any(expression < 0):
                raise ValueError(f"{key} contains negative values")
        
        # Check gene names
        if len(data['gene_names']) != n_genes:
            raise ValueError(
                f"Gene names count mismatch: {len(data['gene_names'])} vs {n_genes}"
            )
        
        self.logger.debug("Data validation passed")
        return True
