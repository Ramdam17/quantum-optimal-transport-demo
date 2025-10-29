#!/usr/bin/env python3
"""
Test Quantum Optimal Transport pipeline on simulated data.

This script demonstrates the complete QOT workflow:
1. Load/generate simulated data for each scenario
2. Compute classical optimal transport
3. Compute quantum optimal transport
4. Compare results and visualize

Usage:
    python scripts/test_qot_pipeline.py --scenario hyperscanning
    python scripts/test_qot_pipeline.py --scenario llm_alignment
    python scripts/test_qot_pipeline.py --scenario genetics
    python scripts/test_qot_pipeline.py --all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.hyperscanning_simulator import HyperscanningSimulator
from src.data.llm_simulator import LLMAlignmentSimulator
from src.data.genetics_simulator import GeneticsSimulator
from src.optimal_transport.classical import OptimalTransport
from src.quantum.qot_algorithms import QuantumOT
from src.quantum.qot_metrics import QuantumMetrics, compare_classical_quantum
from src.visualization.quantum_plots import (
    plot_convergence,
    plot_quantum_vs_classical,
    plot_statevector_bars,
    plot_probability_distribution
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hyperscanning_scenario(seed: int = 42, output_dir: Path = None):
    """
    Test QOT on hyperscanning data.
    
    Parameters
    ----------
    seed : int
        Random seed
    output_dir : Path
        Output directory for plots
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: Hyperscanning (Brain Activity Synchronization)")
    logger.info("=" * 70)
    
    # Generate data
    logger.info("Generating brain activity data...")
    simulator = HyperscanningSimulator(seed=seed)
    data = simulator.generate(
        n_subjects=2,
        n_regions=4,  # Small for quantum feasibility
        n_timepoints=100,
        inter_subject_correlation=0.6
    )
    
    # Extract region activity distributions
    subject1_activity = np.mean(data['subject1'], axis=1)  # Average over time
    subject2_activity = np.mean(data['subject2'], axis=1)
    
    # Normalize to probability distributions
    subject1_dist = subject1_activity / np.sum(subject1_activity)
    subject2_dist = subject2_activity / np.sum(subject2_activity)
    
    logger.info(f"Subject 1 distribution: {subject1_dist}")
    logger.info(f"Subject 2 distribution: {subject2_dist}")
    
    # Classical OT
    logger.info("\nComputing Classical Optimal Transport...")
    start_time = time.time()
    ot = OptimalTransport(method='sinkhorn', reg=0.01)
    classical_result = ot.compute(
        subject1_dist.reshape(-1, 1),
        subject2_dist.reshape(-1, 1)
    )
    classical_time = time.time() - start_time
    classical_result['execution_time'] = classical_time
    
    logger.info(f"Classical OT Cost: {classical_result['cost']:.6f}")
    logger.info(f"Classical OT Time: {classical_time:.3f}s")
    
    # Quantum OT
    logger.info("\nComputing Quantum Optimal Transport (VQE)...")
    start_time = time.time()
    n_qubits = int(np.ceil(np.log2(len(subject1_dist))))
    qot = QuantumOT(
        n_qubits=n_qubits,
        method='vqe',
        max_iterations=30,
        seed=seed
    )
    quantum_result = qot.compute(subject1_dist, subject2_dist)
    quantum_time = time.time() - start_time
    
    logger.info(f"Quantum OT Cost: {quantum_result['cost']:.6f}")
    logger.info(f"Quantum OT Time: {quantum_time:.3f}s")
    logger.info(f"Iterations: {quantum_result['iterations']}")
    logger.info(f"Converged: {quantum_result['success']}")
    
    # Comparison
    logger.info("\nComparison:")
    metrics = QuantumMetrics()
    comparison = metrics.compare_ot_results(classical_result, quantum_result)
    logger.info(f"Cost Difference: {comparison['cost_difference']:.6f}")
    logger.info(f"Relative Error: {comparison['relative_error']:.2f}%")
    logger.info(f"Speedup: {classical_time/quantum_time:.2f}x")
    
    # Visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convergence plot
        plot_convergence(
            quantum_result['convergence_history'],
            save_path=output_dir / 'hyperscanning_convergence.png',
            title='Hyperscanning QOT Convergence',
            show=False
        )
        
        # Comparison plot
        plot_quantum_vs_classical(
            classical_result,
            quantum_result,
            save_path=output_dir / 'hyperscanning_comparison.png',
            show=False
        )
        
        # Final quantum state
        plot_statevector_bars(
            quantum_result['final_state'],
            save_path=output_dir / 'hyperscanning_state.png',
            title='Final Quantum State',
            show=False
        )
        
        logger.info(f"\nPlots saved to {output_dir}")
    
    return {
        'classical': classical_result,
        'quantum': quantum_result,
        'comparison': comparison
    }


def test_llm_alignment_scenario(seed: int = 42, output_dir: Path = None):
    """
    Test QOT on LLM alignment data.
    
    Parameters
    ----------
    seed : int
        Random seed
    output_dir : Path
        Output directory for plots
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: LLM Alignment (Embedding Space Comparison)")
    logger.info("=" * 70)
    
    # Generate data
    logger.info("Generating LLM embedding data...")
    simulator = LLMAlignmentSimulator(seed=seed)
    data = simulator.generate(
        vocab_size=8,  # Small for quantum feasibility
        embedding_dim=4,
        similarity=0.7
    )
    
    # Use vocabulary distribution as probability distributions
    # (frequency of words in each model)
    model1_dist = np.random.dirichlet(np.ones(8), 1)[0]
    model2_dist = np.random.dirichlet(np.ones(8), 1)[0]
    
    # Make them somewhat similar
    model2_dist = 0.7 * model1_dist + 0.3 * model2_dist
    model1_dist = model1_dist / np.sum(model1_dist)
    model2_dist = model2_dist / np.sum(model2_dist)
    
    logger.info(f"Model 1 word distribution shape: {model1_dist.shape}")
    logger.info(f"Model 2 word distribution shape: {model2_dist.shape}")
    
    # Classical OT
    logger.info("\nComputing Classical Optimal Transport...")
    start_time = time.time()
    ot = OptimalTransport(method='sinkhorn', reg=0.01)
    classical_result = ot.compute(
        model1_dist.reshape(-1, 1),
        model2_dist.reshape(-1, 1)
    )
    classical_time = time.time() - start_time
    classical_result['execution_time'] = classical_time
    
    logger.info(f"Classical OT Cost: {classical_result['cost']:.6f}")
    logger.info(f"Classical OT Time: {classical_time:.3f}s")
    
    # Quantum OT
    logger.info("\nComputing Quantum Optimal Transport (QAOA)...")
    start_time = time.time()
    n_qubits = int(np.ceil(np.log2(len(model1_dist))))
    qot = QuantumOT(
        n_qubits=n_qubits,
        method='qaoa',
        max_iterations=30,
        seed=seed
    )
    quantum_result = qot.compute(model1_dist, model2_dist)
    quantum_time = time.time() - start_time
    
    logger.info(f"Quantum OT Cost: {quantum_result['cost']:.6f}")
    logger.info(f"Quantum OT Time: {quantum_time:.3f}s")
    logger.info(f"Iterations: {quantum_result['iterations']}")
    
    # Comparison
    logger.info("\nComparison:")
    comparison = compare_classical_quantum(classical_result, quantum_result)
    logger.info(f"Cost Difference: {comparison['cost_difference']:.6f}")
    logger.info(f"Relative Error: {comparison['relative_error']:.2f}%")
    
    # Visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_convergence(
            quantum_result['convergence_history'],
            save_path=output_dir / 'llm_convergence.png',
            title='LLM Alignment QOT Convergence',
            show=False
        )
        
        plot_quantum_vs_classical(
            classical_result,
            quantum_result,
            save_path=output_dir / 'llm_comparison.png',
            show=False
        )
        
        plot_probability_distribution(
            quantum_result['final_probabilities'],
            save_path=output_dir / 'llm_probabilities.png',
            title='Final Probability Distribution',
            show=False
        )
        
        logger.info(f"\nPlots saved to {output_dir}")
    
    return {
        'classical': classical_result,
        'quantum': quantum_result,
        'comparison': comparison
    }


def test_genetics_scenario(seed: int = 42, output_dir: Path = None):
    """
    Test QOT on genetics data.
    
    Parameters
    ----------
    seed : int
        Random seed
    output_dir : Path
        Output directory for plots
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: Genetics (Gene Expression Comparison)")
    logger.info("=" * 70)
    
    # Generate data
    logger.info("Generating gene expression data...")
    simulator = GeneticsSimulator(seed=seed)
    data = simulator.generate(
        n_genes=8,  # Small for quantum feasibility
        n_samples_per_pop=50,
        n_populations=2,
        population_names=['population_A', 'population_B'],
        n_differential_genes=2,
        differential_magnitude=1.0
    )
    
    # Compute mean expression per gene for each population
    pop1_expression = np.mean(data['population_A_expression'], axis=0)
    pop2_expression = np.mean(data['population_B_expression'], axis=0)
    
    # Normalize to probability distributions
    pop1_dist = pop1_expression / np.sum(pop1_expression)
    pop2_dist = pop2_expression / np.sum(pop2_expression)
    
    logger.info(f"Population 1 expression distribution: {pop1_dist}")
    logger.info(f"Population 2 expression distribution: {pop2_dist}")
    
    # Classical OT
    logger.info("\nComputing Classical Optimal Transport...")
    start_time = time.time()
    ot = OptimalTransport(method='sinkhorn', reg=0.01)
    classical_result = ot.compute(
        pop1_dist.reshape(-1, 1),
        pop2_dist.reshape(-1, 1)
    )
    classical_time = time.time() - start_time
    classical_result['execution_time'] = classical_time
    
    logger.info(f"Classical OT Cost: {classical_result['cost']:.6f}")
    logger.info(f"Classical OT Time: {classical_time:.3f}s")
    
    # Quantum OT
    logger.info("\nComputing Quantum Optimal Transport (VQE)...")
    start_time = time.time()
    n_qubits = int(np.ceil(np.log2(len(pop1_dist))))
    qot = QuantumOT(
        n_qubits=n_qubits,
        method='vqe',
        max_iterations=30,
        optimizer='SLSQP',
        seed=seed
    )
    quantum_result = qot.compute(pop1_dist, pop2_dist)
    quantum_time = time.time() - start_time
    
    logger.info(f"Quantum OT Cost: {quantum_result['cost']:.6f}")
    logger.info(f"Quantum OT Time: {quantum_time:.3f}s")
    logger.info(f"Iterations: {quantum_result['iterations']}")
    
    # Comparison
    logger.info("\nComparison:")
    comparison = compare_classical_quantum(classical_result, quantum_result)
    logger.info(f"Cost Difference: {comparison['cost_difference']:.6f}")
    logger.info(f"Relative Error: {comparison['relative_error']:.2f}%")
    
    # Visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_convergence(
            quantum_result['convergence_history'],
            save_path=output_dir / 'genetics_convergence.png',
            title='Genetics QOT Convergence',
            show=False
        )
        
        plot_quantum_vs_classical(
            classical_result,
            quantum_result,
            save_path=output_dir / 'genetics_comparison.png',
            show=False
        )
        
        plot_statevector_bars(
            quantum_result['final_state'],
            save_path=output_dir / 'genetics_state.png',
            title='Final Quantum State',
            show=False
        )
        
        logger.info(f"\nPlots saved to {output_dir}")
    
    return {
        'classical': classical_result,
        'quantum': quantum_result,
        'comparison': comparison
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test Quantum Optimal Transport on simulated data'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['hyperscanning', 'llm_alignment', 'genetics', 'all'],
        default='all',
        help='Scenario to test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/qot_tests',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    results = {}
    
    # Run scenarios
    if args.scenario in ['hyperscanning', 'all']:
        results['hyperscanning'] = test_hyperscanning_scenario(
            seed=args.seed,
            output_dir=output_dir / 'hyperscanning'
        )
    
    if args.scenario in ['llm_alignment', 'all']:
        results['llm_alignment'] = test_llm_alignment_scenario(
            seed=args.seed,
            output_dir=output_dir / 'llm_alignment'
        )
    
    if args.scenario in ['genetics', 'all']:
        results['genetics'] = test_genetics_scenario(
            seed=args.seed,
            output_dir=output_dir / 'genetics'
        )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    for scenario, result in results.items():
        logger.info(f"\n{scenario.upper()}:")
        logger.info(f"  Classical Cost: {result['classical']['cost']:.6f}")
        logger.info(f"  Quantum Cost: {result['quantum']['cost']:.6f}")
        logger.info(f"  Relative Error: {result['comparison']['relative_error']:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("All tests completed successfully! ✅")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
