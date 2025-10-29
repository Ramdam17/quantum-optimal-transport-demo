#!/usr/bin/env python3
"""
Demo script to generate visualizations using simulated data.

This script demonstrates the OT visualization capabilities with data
from all three scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.genetics_simulator import GeneticsSimulator
from src.data.hyperscanning_simulator import HyperscanningSimulator
from src.data.llm_simulator import LLMAlignmentSimulator
from src.optimal_transport.classical import OptimalTransport
from src.visualization.ot_plots import (plot_cost_matrix,
                                        plot_distributions_1d,
                                        plot_distributions_2d,
                                        plot_ot_comparison,
                                        plot_transport_arrows_2d,
                                        plot_transport_plan)

# Create output directory
output_dir = Path("outputs/figures/demo")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("QUANTUM OPTIMAL TRANSPORT - VISUALIZATION DEMO")
print("=" * 70)


# ============================================================================
# SCENARIO 1: Hyperscanning - Brain Activity Patterns
# ============================================================================
print("\n📊 Scenario 1: Hyperscanning (Brain Activity)")
print("-" * 70)

# Generate brain activity data for two subjects
hyper_sim = HyperscanningSimulator(seed=42)
hyper_data = hyper_sim.generate(
    n_subjects=2, n_regions=10, n_timepoints=100, synchrony=0.7
)

# Extract activity patterns from two ROIs for visualization
subject1_roi1 = hyper_data["subject1"][0, :]  # First ROI, all timepoints
subject2_roi1 = hyper_data["subject2"][0, :]  # First ROI, all timepoints

# 1D visualization
print("  → Generating 1D distribution plot...")
fig1 = plot_distributions_1d(
    subject1_roi1,
    subject2_roi1,
    bins=40,  # Fewer bins to avoid overlap
    title="Brain Activity Distribution - ROI 1",
    save_path=output_dir / "hyperscanning_1d.png",
)
plt.close(fig1)

# 2D visualization using two ROIs with fewer samples
subject1_2d = hyper_data["subject1"][:2, :20].T  # First 2 ROIs, 20 timepoints
subject2_2d = hyper_data["subject2"][:2, :20].T

# NOTE PÉDAGOGIQUE: Pour n=20 points, nous utilisons une régularisation adaptative
n_points_hyper = len(subject1_2d)
adaptive_reg_hyper = 0.01 if n_points_hyper >= 30 else 0.1
print(f"  → Paramètres adaptatifs: n={n_points_hyper}, reg={adaptive_reg_hyper}")

print("  → Generating 2D distribution plot...")
fig2 = plot_distributions_2d(
    subject1_2d,
    subject2_2d,
    title="Brain Activity Patterns - 2 ROIs (20 timepoints)",
    save_path=output_dir / "hyperscanning_2d.png",
)
plt.close(fig2)

# Compute optimal transport with adaptive regularization
print("  → Computing optimal transport...")
ot_solver = OptimalTransport(method="sinkhorn", reg=adaptive_reg_hyper)
ot_result = ot_solver.compute(subject1_2d, subject2_2d)

# Transport plan visualization
print("  → Generating transport plan heatmap...")
fig3 = plot_transport_plan(
    ot_result["transport_plan"],
    title="OT Plan: Subject 1 → Subject 2 (Brain Activity)",
    save_path=output_dir / "hyperscanning_transport_plan.png",
)
plt.close(fig3)

# Cost matrix
print("  → Generating cost matrix...")
fig4 = plot_cost_matrix(
    ot_result["cost_matrix"],
    title="Cost Matrix: Subject 1 vs Subject 2",
    save_path=output_dir / "hyperscanning_cost_matrix.png",
)
plt.close(fig4)

# NOTE: Le threshold est calculé dynamiquement à 10% de la valeur maximale
# du plan de transport pour garantir une visualisation claire des transports
# principaux tout en évitant la surcharge visuelle.
transport_plan_hyper = ot_result["transport_plan"]
nonzero_transport_hyper = transport_plan_hyper[transport_plan_hyper > 1e-10]

if len(nonzero_transport_hyper) > 0:
    max_val_hyper = nonzero_transport_hyper.max()
    mean_val_hyper = nonzero_transport_hyper.mean()

    # Threshold = 10% du max (affiche ~30-50% des transports)
    adaptive_threshold_hyper = max_val_hyper * 0.10

    print(
        f"  → Transport: max={max_val_hyper:.6f}, mean={mean_val_hyper:.6f}, threshold={adaptive_threshold_hyper:.6f}"
    )
else:
    adaptive_threshold_hyper = 1e-8

# Arrow visualization with adaptive threshold
print("  → Generating transport arrows...")
fig5 = plot_transport_arrows_2d(
    subject1_2d,
    subject2_2d,
    transport_plan_hyper,
    threshold=adaptive_threshold_hyper,
    title="Transport Map: Brain Activity Alignment (20 timepoints)",
    save_path=output_dir / "hyperscanning_arrows.png",
)
plt.close(fig5)

print(f"  ✓ OT Cost: {ot_result['cost']:.6f}")
print(f"  ✓ Transport plan shape: {ot_result['transport_plan'].shape}")


# ============================================================================
# SCENARIO 2: LLM Alignment - Embedding Spaces
# ============================================================================
print("\n📊 Scenario 2: LLM Alignment (Embedding Spaces)")
print("-" * 70)

# Generate embedding data
llm_sim = LLMAlignmentSimulator(seed=42)
llm_data = llm_sim.generate(
    vocab_size=100,
    embed_dim=2,  # 2D for visualization
    n_models=2,
    alignment_level=0.6,
    semantic_clusters=3,
)

# Use 30 words for better visualization clarity
model1_emb = llm_data["model_A_embeddings"][:30]  # First 30 words
model2_emb = llm_data["model_B_embeddings"][:30]

# NOTE PÉDAGOGIQUE: Pour n=30 points, nous utilisons une régularisation adaptative
n_points_llm = len(model1_emb)
adaptive_reg_llm = 0.01 if n_points_llm >= 30 else 0.1
print(f"  → Paramètres adaptatifs: n={n_points_llm}, reg={adaptive_reg_llm}")

# 2D visualization
print("  → Generating 2D embedding plot...")
fig6 = plot_distributions_2d(
    model1_emb,
    model2_emb,
    title="LLM Embedding Spaces (30 words)",
    save_path=output_dir / "llm_2d.png",
)
plt.close(fig6)

# Compute OT with adaptive regularization
print("  → Computing optimal transport...")
ot_solver_llm = OptimalTransport(method="sinkhorn", reg=adaptive_reg_llm)
ot_result_llm = ot_solver_llm.compute(model1_emb, model2_emb)

# Transport plan visualization
print("  → Generating transport plan heatmap...")
fig7 = plot_transport_plan(
    ot_result_llm["transport_plan"],
    title="OT Plan: Model A → Model B (Word Embeddings)",
    save_path=output_dir / "llm_transport_plan.png",
)
plt.close(fig7)

# Cost matrix
print("  → Generating cost matrix...")
fig8 = plot_cost_matrix(
    ot_result_llm["cost_matrix"],
    title="Cost Matrix: Model A vs Model B",
    save_path=output_dir / "llm_cost_matrix.png",
)
plt.close(fig8)

# NOTE: Le threshold est calculé dynamiquement à 10% de la valeur maximale
# du plan de transport pour garantir une visualisation claire des transports
# principaux tout en évitant la surcharge visuelle.
transport_plan_llm = ot_result_llm["transport_plan"]
nonzero_transport_llm = transport_plan_llm[transport_plan_llm > 1e-10]

if len(nonzero_transport_llm) > 0:
    max_val_llm = nonzero_transport_llm.max()
    mean_val_llm = nonzero_transport_llm.mean()

    # Threshold = 10% du max (affiche ~30-50% des transports)
    adaptive_threshold_llm = max_val_llm * 0.10

    print(
        f"  → Transport: max={max_val_llm:.6f}, mean={mean_val_llm:.6f}, threshold={adaptive_threshold_llm:.6f}"
    )
else:
    adaptive_threshold_llm = 1e-8

# Transport arrows with adaptive threshold
print("  → Generating transport arrows...")
fig9 = plot_transport_arrows_2d(
    model1_emb,
    model2_emb,
    transport_plan_llm,
    threshold=adaptive_threshold_llm,
    title="Embedding Alignment: Model A → Model B (30 words)",
    save_path=output_dir / "llm_arrows.png",
)
plt.close(fig9)

print(f"  ✓ OT Cost: {ot_result_llm['cost']:.6f}")
print(f"  ✓ Alignment quality: {llm_data['alignment_level']:.2f}")


# ============================================================================
# SCENARIO 3: Genetics - Gene Expression
# ============================================================================
print("\n📊 Scenario 3: Genetics (Gene Expression)")
print("-" * 70)

# Generate gene expression data
gen_sim = GeneticsSimulator(seed=42)
gen_data = gen_sim.generate(
    n_genes=50,
    n_samples_per_population=80,
    differential_genes_ratio=0.3,
    differential_magnitude=2.0,  # Stronger differences
    biological_noise=0.3,  # Add some noise
)

# Use 30 samples for better visualization clarity
pop1 = gen_data["population_A_expression"][:30]  # First 30 samples from population A
pop2 = gen_data["population_B_expression"][:30]  # First 30 samples from population B

# Use first two genes for 2D visualization and NORMALIZE
pop1_2d = pop1[:, :2]
pop2_2d = pop2[:, :2]

# Normalize to prevent numerical issues (gene expression values are large)
scaler = StandardScaler()
pop1_2d_scaled = scaler.fit_transform(pop1_2d)
pop2_2d_scaled = scaler.transform(pop2_2d)

# NOTE PÉDAGOGIQUE: Pour n=30 points, nous utilisons une régularisation adaptative
n_points_gen = len(pop1_2d_scaled)
adaptive_reg_gen = 0.01 if n_points_gen >= 30 else 0.1
print(f"  → Paramètres adaptatifs: n={n_points_gen}, reg={adaptive_reg_gen}")

print("  → Generating 2D gene expression plot...")
fig8 = plot_distributions_2d(
    pop1_2d_scaled,  # Use scaled data for visualization
    pop2_2d_scaled,
    title="Gene Expression: Population 1 vs 2 (2 genes, 30 samples, normalized)",
    save_path=output_dir / "genetics_2d.png",
)
plt.close(fig8)

# Compute OT with scaled data and adaptive regularization
print("  → Computing optimal transport with normalized data...")
ot_solver_genetics = OptimalTransport(method="sinkhorn", reg=adaptive_reg_gen)
ot_result_gen = ot_solver_genetics.compute(pop1_2d_scaled, pop2_2d_scaled)

# Transport plan visualization
print("  → Generating transport plan heatmap...")
fig10 = plot_transport_plan(
    ot_result_gen["transport_plan"],
    title="OT Plan: Population A → Population B",
    save_path=output_dir / "genetics_transport_plan.png",
)
plt.close(fig10)

# Cost matrix
print("  → Generating cost matrix...")
fig11 = plot_cost_matrix(
    ot_result_gen["cost_matrix"],
    title="Cost Matrix: Population A vs B",
    save_path=output_dir / "genetics_cost_matrix.png",
)
plt.close(fig11)

# NOTE: Le threshold est calculé dynamiquement à 10% de la valeur maximale
# du plan de transport pour garantir une visualisation claire des transports
# principaux tout en évitant la surcharge visuelle.
transport_plan_gen = ot_result_gen["transport_plan"]
nonzero_transport_gen = transport_plan_gen[transport_plan_gen > 1e-10]

if len(nonzero_transport_gen) > 0:
    max_val_gen = nonzero_transport_gen.max()
    mean_val_gen = nonzero_transport_gen.mean()

    # Threshold = 10% du max (affiche ~30-50% des transports)
    adaptive_threshold_gen = max_val_gen * 0.10

    print(
        f"  → Transport: max={max_val_gen:.6f}, mean={mean_val_gen:.6f}, threshold={adaptive_threshold_gen:.6f}"
    )
else:
    adaptive_threshold_gen = 1e-8

# Transport arrows with adaptive threshold
print("  → Generating transport arrows...")
fig12 = plot_transport_arrows_2d(
    pop1_2d_scaled,
    pop2_2d_scaled,
    transport_plan_gen,
    threshold=adaptive_threshold_gen,
    title="Gene Expression Transport: Population A → B (30 samples, normalized)",
    save_path=output_dir / "genetics_arrows.png",
)
plt.close(fig12)

print(f"  ✓ OT Cost: {ot_result_gen['cost']:.6f}")
print(f"  ✓ Differential genes: {len(gen_data['differential_genes'])}")


# ============================================================================
# COMPARISON ACROSS METHODS
# ============================================================================
print("\n📊 Method Comparison")
print("-" * 70)

# Compare Sinkhorn vs Exact on small dataset
print("  → Computing OT with different methods...")
small_source = np.random.randn(20, 2)
small_target = np.random.randn(25, 2)

results = {}

# Sinkhorn with different regularizations
for reg in [0.01, 0.05, 0.1]:
    ot_sinkhorn = OptimalTransport(method="sinkhorn", reg=reg)
    result = ot_sinkhorn.compute(small_source, small_target)
    results[f"Sinkhorn (λ={reg})"] = result

# Exact OT
ot_exact = OptimalTransport(method="exact")
result_exact = ot_exact.compute(small_source, small_target)
results["Exact OT"] = result_exact

print("  → Generating comparison plot...")
fig10 = plot_ot_comparison(
    results,
    metric="cost",
    title="OT Cost Comparison: Different Methods",
    save_path=output_dir / "method_comparison.png",
)
plt.close(fig10)

# Print comparison
print("\n  Results:")
for method, result in results.items():
    print(f"    {method:20s} - Cost: {result['cost']:.6f}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION DEMO COMPLETE! 🎉")
print("=" * 70)
print("\n📁 All figures saved to: {output_dir.absolute()}")
print("\nGenerated visualizations:")
print("  • Hyperscanning: 5 figures (1D, 2D, transport plan, cost matrix, arrows)")
print("  • LLM Alignment: 4 figures (2D, transport plan, cost matrix, arrows)")
print("  • Genetics: 4 figures (2D, transport plan, cost matrix, arrows)")
print("  • Method Comparison: 1 figure (bar chart)")
print(f"\n  Total: 14 figures")
print("\n✨ Check the output directory to see your visualizations!")
print("=" * 70)
