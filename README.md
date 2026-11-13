# Quantum Optimal Transport — a Hands-On Course

A hands-on journey from **classical optimal transport** to its **quantum generalization**, quantum
optimal transport (QOT) — ending in a controlled test. The phase-locking value (PLV) is a standard
measure of how steadily two signals keep the same phase difference, and the test asks *how far* a
coupling measure built from QOT reads past it, with a classical statistic beside it at each test.

The course is built **one concept per notebook**: short, focused "bricks" build progressively to a
"synthesis" notebook, so each idea lands before the next arrives. **76 notebooks across 6 modules**,
sized for ~1-hour sessions and a mixed-background audience, and meant to be worked through with an AI
tutor alongside.

## What you'll build

Starting from qubits and density matrices, you assemble — piece by piece — the machinery of optimal
transport on quantum states: the coupling semidefinite program, the operator Sinkhorn iteration, the
Bures–Wasserstein bridge, and the information-geometric picture that ties them together. The classical
↔ quantum **dictionary** lives in one notebook, `notebooks/dictionary.ipynb`; each notebook shows only the rows it earns. The theory closes with a research-grade
capstone, and the last module asks whether a real quantum processor could resolve the capstone's
results.

## The capstone — how far quantum optimal transport reads past PLV

The capstone (`11` to `18` in `notebooks/04_QuantumOptimalTransport/`) does not hand-wave its
question: can a quantum coupling measure read something the phase-locking value cannot?

Three words carry the answer. Write Δ for the phase difference between two signals. PLV is the length
of the average of `e^{iΔ}`, and that average is the *first harmonic* of the phase difference. The
*second harmonic* is the average of `e^{2iΔ}`, which PLV discards; it tells, for instance, whether
the phase difference gathers around two opposite values instead of one. An *embedding* is the recipe
that writes the two signals into a quantum state, and it decides which harmonics the state carries.

The capstone builds two synthetic ensembles, sets of sampled phase pairs, with the same first
harmonic, hence **identical PLV by construction**, and a different second harmonic. It then simulates
a pair of coupled oscillators whose second harmonic has a coupling strength of its own. Each quantum
measure is scored beside a classical statistic of the same harmonics. The capstone reaches four
conclusions:

1. When its embedding carries the second harmonic, a quantum coupling measure **reads structure that
   PLV cannot**. On the simplest embedding, which carries the first harmonic only, it follows PLV.
2. It reads **no further than a classical statistic of the same two harmonics**, and with equal
   weights on the harmonics it needs more data to do so.
3. On the *relative phase* of the two harmonics, that is, how the second is shifted against the
   first, the quantum measures computed in the course read **less** than a classical statistic: they
   are blind to the direction of the shift.
4. What quantum optimal transport brings is a **unified geometric framework**: one state, one distance
   from independence, and the embedding as the single dial that sets what the coupling measure reads.
   On the synthetic systems of the capstone that is a unification, not a gain in sensitivity.

The last module then measures how finely a real IBM processor reads a mutual information, on a
two-qubit probe, and sets the capstone's numbers against that resolution. It runs offline on a noisy
device model by default, and on the real device if you switch it on.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

All deliverables (code, notebooks, summaries, glossary, the classical ↔ quantum dictionary, `notebooks/dictionary.ipynb`) are in
English.

## How the course is structured

The six module folders hold plain `.ipynb` files (not a Jupyter Book), numbered within each module as
`NN_snake_title.ipynb`:

- **`00_GettingStarted`** — environment setup, and what optimal transport is.
- **`01_QuantumFoundations`** — qubits and complex amplitudes, the inner product, gates (Pauli,
  Hadamard, phase/rotations, CNOT) and expectation values, density matrices, composite systems, channels.
- **`02_InformationAndGeometry`** — classical & quantum information theory, the Fisher–Rao and Bures
  geometries.
- **`03_ClassicalOptimalTransport`** — Monge/Kantorovich, Wasserstein, Sinkhorn, the Gaussian/Bures
  bridge.
- **`04_QuantumOptimalTransport`** — the quantum coupling SDP, quantum Sinkhorn, the QOT "zoo", and the
  capstone that tests the coupling measure against PLV and its classical neighbours.
- **`05_RealQuantumHardware`** — run the course's results (Born rule, tomography, quantum mutual
  information, Bures distance) on a real IBM QPU through the runtime primitives, and set the capstone's
  numbers against the resolution of the device.

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

Module `05_RealQuantumHardware` is where this happens in practice: each notebook has a
`USE_HARDWARE` switch — `False` (the default) runs offline on a noisy fake backend; `True` runs on
a real QPU once your account is set up (`save_account`, or a git-ignored `.env` with
`IBM_QUANTUM_TOKEN`).

## License

This repository is released under a dual license:

- **Code** (`src/`, `scripts/`, `tests/`) — MIT License. See [`LICENSE`](LICENSE).
- **Course content** (`notebooks/`, `docs/`) — Creative Commons Attribution 4.0 International
  (CC BY 4.0). See [`LICENSE-CONTENT`](LICENSE-CONTENT).

© 2026 Rémy Ramadour.

## Contributing

Contributions — fixes, sharper explanations, new exercises, library extensions — are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).
