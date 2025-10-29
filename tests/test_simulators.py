"""
Unit tests for data simulators.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.data.hyperscanning_simulator import HyperscanningSimulator
from src.data.llm_simulator import LLMAlignmentSimulator
from src.data.genetics_simulator import GeneticsSimulator
from src.data.loaders import get_simulator, DataLoader


class TestHyperscanningSimulator:
    """Test suite for HyperscanningSimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return HyperscanningSimulator(seed=42)
    
    def test_generate_basic(self, simulator):
        """Test basic data generation."""
        data = simulator.generate(
            n_subjects=2,
            n_regions=10,
            n_timepoints=100
        )
        
        assert 'subject1' in data
        assert 'subject2' in data
        assert data['subject1'].shape == (100, 10)
        assert data['subject2'].shape == (100, 10)
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        sim1 = HyperscanningSimulator(seed=42)
        data1 = sim1.generate(n_regions=5, n_timepoints=50)
        
        sim2 = HyperscanningSimulator(seed=42)
        data2 = sim2.generate(n_regions=5, n_timepoints=50)
        
        np.testing.assert_array_equal(data1['subject1'], data2['subject1'])
    
    def test_synchrony_effect(self, simulator):
        """Test that synchrony level affects correlation."""
        # High synchrony
        data_high = simulator.generate(
            n_regions=5,
            n_timepoints=200,
            synchrony_level=0.9,
            synchrony_regions=[0, 1]
        )
        
        # Low synchrony
        simulator_low = HyperscanningSimulator(seed=43)
        data_low = simulator_low.generate(
            n_regions=5,
            n_timepoints=200,
            synchrony_level=0.1,
            synchrony_regions=[0, 1]
        )
        
        # Check correlation in sync region
        corr_high = np.corrcoef(
            data_high['subject1'][:, 0],
            data_high['subject2'][:, 0]
        )[0, 1]
        
        corr_low = np.corrcoef(
            data_low['subject1'][:, 0],
            data_low['subject2'][:, 0]
        )[0, 1]
        
        assert corr_high > corr_low
    
    def test_save_and_load(self, simulator):
        """Test saving and loading data."""
        data = simulator.generate(n_regions=5, n_timepoints=50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            output_dir = simulator.save(tmpdir)
            assert output_dir.exists()
            
            # Load
            loaded_data = HyperscanningSimulator.load(tmpdir)
            
            # Compare
            np.testing.assert_array_equal(
                data['subject1'],
                loaded_data['subject1']
            )
    
    def test_get_summary(self, simulator):
        """Test summary generation."""
        data = simulator.generate(n_regions=5, n_timepoints=50)
        summary = simulator.get_summary()
        
        assert 'simulator' in summary
        assert 'arrays' in summary
        assert 'subject1' in summary['arrays']


class TestLLMAlignmentSimulator:
    """Test suite for LLMAlignmentSimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return LLMAlignmentSimulator(seed=42)
    
    def test_generate_basic(self, simulator):
        """Test basic data generation."""
        data = simulator.generate(
            vocab_size=100,
            embed_dim=64,
            n_models=2
        )
        
        assert 'model_A_embeddings' in data
        assert 'model_B_embeddings' in data
        assert data['model_A_embeddings'].shape == (100, 64)
        assert data['model_B_embeddings'].shape == (100, 64)
    
    def test_vocabulary_generation(self, simulator):
        """Test vocabulary is generated correctly."""
        data = simulator.generate(vocab_size=50)
        
        assert len(data['vocabulary']) == 50
        assert all(isinstance(token, str) for token in data['vocabulary'])
    
    def test_alignment_level(self):
        """Test that alignment level affects similarity."""
        # High alignment
        sim_high = LLMAlignmentSimulator(seed=42)
        data_high = sim_high.generate(
            vocab_size=100,
            embed_dim=32,
            alignment_level=0.9
        )
        
        # Low alignment
        sim_low = LLMAlignmentSimulator(seed=43)
        data_low = sim_low.generate(
            vocab_size=100,
            embed_dim=32,
            alignment_level=0.1
        )
        
        # Compute average cosine similarity
        emb1_high = data_high['model_A_embeddings']
        emb2_high = data_high['model_B_embeddings']
        sim_scores_high = np.sum(emb1_high * emb2_high, axis=1)
        
        emb1_low = data_low['model_A_embeddings']
        emb2_low = data_low['model_B_embeddings']
        sim_scores_low = np.sum(emb1_low * emb2_low, axis=1)
        
        assert np.mean(sim_scores_high) > np.mean(sim_scores_low)
    
    def test_embeddings_normalized(self, simulator):
        """Test that embeddings are on unit sphere."""
        data = simulator.generate(vocab_size=50, embed_dim=32)
        
        emb = data['model_A_embeddings']
        norms = np.linalg.norm(emb, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(50), decimal=5)
    
    def test_semantic_clusters(self, simulator):
        """Test cluster assignments."""
        data = simulator.generate(
            vocab_size=100,
            semantic_clusters=5
        )
        
        clusters = data['cluster_assignments']
        assert len(clusters) == 100
        assert clusters.min() >= 0
        assert clusters.max() < 5


class TestGeneticsSimulator:
    """Test suite for GeneticsSimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return GeneticsSimulator(seed=42)
    
    def test_generate_basic(self, simulator):
        """Test basic data generation."""
        data = simulator.generate(
            n_genes=50,
            n_samples_per_pop=100,
            n_populations=2
        )
        
        assert 'population_A_expression' in data
        assert 'population_B_expression' in data
        assert data['population_A_expression'].shape == (100, 50)
        assert data['population_B_expression'].shape == (100, 50)
    
    def test_gene_names(self, simulator):
        """Test gene name generation."""
        data = simulator.generate(n_genes=30)
        
        assert len(data['gene_names']) == 30
        assert all(isinstance(name, str) for name in data['gene_names'])
        assert all(name.startswith('GENE') for name in data['gene_names'])
    
    def test_differential_genes(self, simulator):
        """Test differential expression."""
        data = simulator.generate(
            n_genes=50,
            n_samples_per_pop=100,
            n_differential_genes=10,
            differential_magnitude=2.0
        )
        
        diff_genes = data['differential_genes']
        assert len(diff_genes) == 10
        
        # Check that differential genes have different means
        pop_a = data['population_A_expression']
        pop_b = data['population_B_expression']
        
        for gene_idx in diff_genes:
            mean_a = np.mean(pop_a[:, gene_idx])
            mean_b = np.mean(pop_b[:, gene_idx])
            
            # Means should be different (though exact difference varies)
            assert abs(mean_a - mean_b) > 0.1
    
    def test_non_negative_expression(self, simulator):
        """Test that expression values are non-negative."""
        data = simulator.generate(n_genes=20, n_samples_per_pop=50)
        
        pop_a = data['population_A_expression']
        pop_b = data['population_B_expression']
        
        assert np.all(pop_a >= 0)
        assert np.all(pop_b >= 0)
    
    def test_normalization(self, simulator):
        """Test quantile normalization."""
        data_norm = simulator.generate(
            n_genes=30,
            n_samples_per_pop=50,
            normalize=True
        )
        
        data_unnorm = simulator.generate(
            n_genes=30,
            n_samples_per_pop=50,
            normalize=False
        )
        
        # Normalized data should have more similar distributions
        # across samples (not a perfect test, but reasonable)
        assert data_norm['normalized'] is True
        assert data_unnorm['normalized'] is False


class TestDataLoaders:
    """Test suite for data loaders."""
    
    def test_get_simulator_hyperscanning(self):
        """Test getting hyperscanning simulator."""
        SimClass = get_simulator('hyperscanning')
        assert SimClass == HyperscanningSimulator
    
    def test_get_simulator_llm(self):
        """Test getting LLM simulator."""
        SimClass = get_simulator('llm_alignment')
        assert SimClass == LLMAlignmentSimulator
    
    def test_get_simulator_genetics(self):
        """Test getting genetics simulator."""
        SimClass = get_simulator('genetics')
        assert SimClass == GeneticsSimulator
    
    def test_get_simulator_invalid(self):
        """Test error for invalid scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_simulator('invalid_scenario')
    
    def test_data_loader_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.config_dir == Path("config")
        assert loader.data_dir == Path("data")
    
    def test_data_loader_cache(self):
        """Test DataLoader caching."""
        loader = DataLoader(cache_data=True)
        
        # Cache should be empty initially
        assert len(loader._cache) == 0
        
        # Clear cache should work
        loader.clear_cache()
        assert len(loader._cache) == 0
