"""
LLM alignment simulator for embedding space comparison.

This module simulates embedding spaces from different language models
with controllable similarity/divergence patterns.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from src.data.base_simulator import BaseSimulator
from src.utils.helpers import validate_array, normalize_distribution


class LLMAlignmentSimulator(BaseSimulator):
    """
    Simulate LLM embedding spaces for alignment comparison.
    
    This simulator generates two embedding spaces with controllable
    similarity patterns, useful for studying model alignment.
    
    Parameters
    ----------
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    **kwargs
        Additional simulator parameters
        
    Examples
    --------
    >>> simulator = LLMAlignmentSimulator(seed=42)
    >>> data = simulator.generate(vocab_size=1000, embed_dim=128)
    >>> print(data['model_A_embeddings'].shape)
    (1000, 128)
    """
    
    def generate(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        n_models: int = 2,
        model_names: Optional[List[str]] = None,
        alignment_level: float = 0.7,
        semantic_clusters: int = 10,
        cluster_separation: float = 2.0,
        shared_concepts: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate simulated LLM embedding spaces.
        
        Parameters
        ----------
        vocab_size : int, optional
            Number of tokens/words, by default 1000
        embed_dim : int, optional
            Embedding dimensionality, by default 128
        n_models : int, optional
            Number of models to compare, by default 2
        model_names : Optional[List[str]], optional
            Names for models, by default None (['model_A', 'model_B'])
        alignment_level : float, optional
            How aligned models are (0-1), by default 0.7
        semantic_clusters : int, optional
            Number of semantic clusters, by default 10
        cluster_separation : float, optional
            Distance between clusters, by default 2.0
        shared_concepts : float, optional
            Proportion of shared vocabulary (0-1), by default 0.8
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - '{model}_embeddings': Embedding arrays for each model
            - 'vocabulary': List of token strings
            - 'cluster_assignments': Cluster labels for tokens
            - 'alignment_matrix': Pairwise alignment scores
            - Parameters used for generation
        """
        self.logger.info(
            f"Generating LLM embeddings: {n_models} models, "
            f"vocab_size={vocab_size}, embed_dim={embed_dim}"
        )
        
        # Default model names
        if model_names is None:
            model_names = [f"model_{chr(65+i)}" for i in range(n_models)]
        
        if len(model_names) != n_models:
            raise ValueError(f"Need {n_models} model names, got {len(model_names)}")
        
        # Generate vocabulary
        vocabulary = [f"token_{i:04d}" for i in range(vocab_size)]
        
        # Assign tokens to semantic clusters
        cluster_assignments = np.random.randint(0, semantic_clusters, vocab_size)
        
        # Generate cluster centers in embedding space
        cluster_centers = np.random.randn(semantic_clusters, embed_dim)
        cluster_centers *= cluster_separation
        
        # Generate embeddings for each model
        model_embeddings = {}
        
        for model_idx, model_name in enumerate(model_names):
            embeddings = np.zeros((vocab_size, embed_dim))
            
            for token_idx in range(vocab_size):
                cluster_id = cluster_assignments[token_idx]
                cluster_center = cluster_centers[cluster_id]
                
                # Determine if this token is shared across models
                is_shared = np.random.rand() < shared_concepts
                
                if is_shared and model_idx > 0:
                    # Use similar embedding to first model with some noise
                    first_model_key = f"{model_names[0]}_embeddings"
                    base_embedding = model_embeddings[first_model_key][token_idx]
                    noise = np.random.randn(embed_dim) * (1 - alignment_level)
                    embeddings[token_idx] = (
                        alignment_level * base_embedding +
                        (1 - alignment_level) * (cluster_center + noise)
                    )
                else:
                    # Generate new embedding around cluster center
                    noise = np.random.randn(embed_dim)
                    embeddings[token_idx] = cluster_center + noise
            
            # Normalize embeddings to unit sphere (common in LLMs)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
            
            model_embeddings[f"{model_name}_embeddings"] = embeddings
        
        # Compute alignment matrix (cosine similarity between models)
        if n_models == 2:
            emb1 = model_embeddings[f"{model_names[0]}_embeddings"]
            emb2 = model_embeddings[f"{model_names[1]}_embeddings"]
            
            # Cosine similarity for each token pair
            alignment_matrix = emb1 @ emb2.T
        else:
            alignment_matrix = None
        
        # Store data
        self.data = {
            **model_embeddings,
            'vocabulary': vocabulary,
            'cluster_assignments': cluster_assignments,
            'alignment_matrix': alignment_matrix,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'n_models': n_models,
            'model_names': model_names,
            'alignment_level': alignment_level,
            'semantic_clusters': semantic_clusters,
            'cluster_separation': cluster_separation,
            'shared_concepts': shared_concepts
        }
        
        # Validate
        self.validate(self.data)
        
        self.logger.info("LLM embedding data generated successfully")
        return self.data
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated LLM embedding data.
        
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
        model_names = data.get('model_names', [])
        vocab_size = data.get('vocab_size')
        embed_dim = data.get('embed_dim')
        
        # Check each model's embeddings
        for model_name in model_names:
            key = f"{model_name}_embeddings"
            if key not in data:
                raise ValueError(f"Missing embeddings for model: {model_name}")
            
            validate_array(
                data[key],
                expected_shape=(vocab_size, embed_dim),
                name=key
            )
        
        # Check vocabulary
        if len(data['vocabulary']) != vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: {len(data['vocabulary'])} vs {vocab_size}"
            )
        
        # Check cluster assignments
        validate_array(
            data['cluster_assignments'],
            expected_shape=(vocab_size,),
            name="cluster_assignments"
        )
        
        self.logger.debug("Data validation passed")
        return True
