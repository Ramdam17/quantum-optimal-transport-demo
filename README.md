# Quantum Optimal Transport — a Hands-On Course

A hands-on journey from **classical optimal transport** to its **quantum generalization** — ending in
a controlled demonstration of *when* quantum optimal transport reads coupling structure that the
phase-locking value (PLV) cannot.

The course is built **one concept per notebook**: short, focused "bricks" build progressively to a
"synthesis" notebook, so each idea lands before the next arrives. **57 notebooks across 5 modules**,
sized for ~1-hour sessions and a mixed-background audience, and meant to be worked through with an AI
tutor alongside.

## What you'll build

Starting from qubits and density matrices, you assemble — piece by piece — the machinery of optimal
transport on quantum states: the coupling semidefinite program, the operator Sinkhorn iteration, the
Bures–Wasserstein bridge, and the information-geometric picture that ties them together. The classical
↔ quantum **dictionary** grows row by row through every module. The course closes with a research-grade
capstone.

## The capstone — when quantum OT beats PLV

The final module (`notebooks/04_QuantumOptimalTransport/`) does not hand-wave the headline question.
It builds two synthetic oscillator dyads with **identical PLV by construction** but different
higher-order phase structure, and shows that a richer-embedding quantum coupling measure **separates
them** where PLV — and the naive quantum embedding — are blind (a controlled, significant separation
against a matched null). The accounting is honest: a classical higher-order statistic could also detect
this difference, so the value of quantum optimal transport here is the **unified geometric framework**
— one geometry, one pair of measures, with the embedding as the single dial that sets what the coupling
measure can see — not a claim of unique detecting power.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

All deliverables (code, notebooks, summaries, glossary, the classical ↔ quantum dictionary) are in
English.

## How the course is structured

Five module folders — plain `.ipynb` (not Jupyter Book), per-module numbering `NN_snake_title.ipynb`:

- **`00_GettingStarted`** — environment setup, and what optimal transport is.
- **`01_QuantumFoundations`** — qubits, density matrices, composite systems, channels.
- **`02_InformationAndGeometry`** — classical & quantum information theory, the Fisher–Rao and Bures
  geometries.
- **`03_ClassicalOptimalTransport`** — Monge/Kantorovich, Wasserstein, Sinkhorn, the Gaussian/Bures
  bridge.
- **`04_QuantumOptimalTransport`** — the quantum coupling SDP, quantum Sinkhorn, the QOT "zoo", and the
  A+B+C capstone.

Each notebook declares its prerequisites and builds on the ones before it. To find which notebook
covers a concept (and what it builds on), read **`llms.txt`** or ask your assistant.

## Using this course with an AI tutor

Most learners follow this course with an AI assistant (Claude, ChatGPT, Gemini) open alongside — and
that's encouraged. To get the most out of it:

- Point your assistant at **`AGENTS.md`** — it tells the assistant how to *tutor* you (guide and hint,
  rather than hand over answers).
- Ask it to **explain the intuition** first, to **quiz you**, and to **check your "Your turn"** attempts
  *after* you've tried them yourself. The struggle is where the learning happens.
- Use **`llms.txt`** (or ask your assistant) to find which notebook covers a concept and what it builds
  on.

Treat the assistant as a patient tutor, not an answer key — you'll learn far more.

## Hardware & credentials

Real IBM Quantum hardware is optional and the course is simulator-first. Never hardcode an API token.
Run `QiskitRuntimeService.save_account(...)` once locally (or put the token in a git-ignored `.env`);
notebooks then call `QiskitRuntimeService()` with no token.

## License

This repository is released under a dual license:

- **Code** (`src/`, `scripts/`, `tests/`) — MIT License. See [`LICENSE`](LICENSE).
- **Course content** (`notebooks/`, `docs/`) — Creative Commons Attribution 4.0 International
  (CC BY 4.0). See [`LICENSE-CONTENT`](LICENSE-CONTENT).

© 2026 Rémy Ramadour.

## Contributing

Contributions — fixes, sharper explanations, new exercises, library extensions — are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).
