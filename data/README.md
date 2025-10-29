# Data Directory

This directory contains simulated data for the three scenarios:
- **Hyperscanning**: Brain activity data from dyadic interactions
- **LLM Alignment**: Embedding spaces from different language models
- **Genetics**: Gene expression profiles from different populations

## Generating Data

To generate data for all scenarios, run:

```bash
poetry run python scripts/generate_data.py --all
```

Or generate data for a specific scenario:

```bash
poetry run python scripts/generate_data.py --scenario hyperscanning
poetry run python scripts/generate_data.py --scenario llm_alignment
poetry run python scripts/generate_data.py --scenario genetics
```

For more options (custom seed, output directory, etc.):

```bash
poetry run python scripts/generate_data.py --help
```

## Data Format

All data is stored in `.npz` format (compressed NumPy arrays) for efficient storage and loading.

### Hyperscanning Data Structure
- `subject1.npy`: Brain activity for subject 1 (shape: `[n_timepoints, n_regions]`)
- `subject2.npy`: Brain activity for subject 2 (shape: `[n_timepoints, n_regions]`)
- `metadata.json`: Parameters used for simulation

### LLM Alignment Data Structure
- `model1_embeddings.npy`: Embedding space for model 1 (shape: `[vocab_size, embed_dim]`)
- `model2_embeddings.npy`: Embedding space for model 2 (shape: `[vocab_size, embed_dim]`)
- `vocabulary.json`: Shared vocabulary
- `metadata.json`: Parameters used for simulation

### Genetics Data Structure
- `population1_expression.npy`: Gene expression for population 1 (shape: `[n_samples, n_genes]`)
- `population2_expression.npy`: Gene expression for population 2 (shape: `[n_samples, n_genes]`)
- `gene_names.json`: Gene identifiers
- `metadata.json`: Parameters used for simulation

## Note

⚠️ **This directory is gitignored** to avoid committing large data files to the repository.
Users should generate their own data using the provided scripts.
