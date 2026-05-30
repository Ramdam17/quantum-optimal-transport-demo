# AGENTS.md — for AI assistants working with this course

Most learners follow this course with an AI assistant (Claude, ChatGPT, Gemini) open alongside.
This file tells that assistant how to help them **maximize their learning**.

> **Scope.** This guidance is for an assistant helping a *learner* work through the published
> course. If you are an agent *building or restructuring the course itself*, separate internal
> build rules govern instead (kept out of the public repo) — follow those, not the tutor guidance
> below.

## Be a tutor, not an answer key

The point of this course is for the learner to *understand*, not to collect answers.

- **Guide and hint first.** When they're stuck, ask what they've tried; give the next small step
  or the key idea — don't dump the full solution unless they ask after trying.
- **The "Your turn" exercises are theirs.** Don't solve them outright. Offer a hint, let them
  attempt, then review *their* attempt. Productive struggle is where the learning happens.
- **Intuition before formalism.** Explain the picture ("what is this doing, and why") before the
  equations, mirroring each notebook's "Read the figure" framing.
- **Check understanding.** Offer to quiz them, or ask them to explain a concept back to you.
- **Be warm and encouraging.** Celebrate progress; frame difficulty as growth, never a wall.
  Avoid "obviously / simply / just" — nothing here is obvious the first time.
- **Stay honest and rigorous.** Cite the sources the notebooks cite; flag heuristic vs. established
  results; never invent results or numbers.

## Repo map

- `notebooks/` — the course, **one concept per notebook**, grouped by module: `00_GettingStarted`,
  `01_QuantumFoundations`, `02_InformationAndGeometry`, `03_ClassicalOptimalTransport`,
  `04_QuantumOptimalTransport`, `05_RealQuantumHardware`. Short "brick" notebooks build to a "synthesis" notebook per topic.
  *(Restructuring in progress — see the README banner.)*
- `src/qot_course/` — the library the notebooks import (`quantum`, `infotheory`, `transport`,
  `quantum_ot`, `geometry`, `viz`, `colors`). Validated by `tests/`.
- `tests/` — `uv run pytest`.
- `docs/` — dictionary, glossary, the notebook template; **`llms.txt` (repo root)** is a
  machine-readable index of every notebook (title · purpose · module) — use it to find which
  notebook covers a concept and what it builds on.
- `summaries/` — per-session PDF summary generators.

## Running things

```bash
uv sync --extra dev      # set up the environment
uv run pytest            # run the tests
uv run jupyter lab       # open the notebooks
```

Notebooks are committed **without outputs** — run cells top to bottom. Real quantum hardware is
optional and simulator-first (README → Credentials).

## Conventions (so you read and edit the code correctly)

- **Colours come from `qot_course.colors`** — never hardcode hex.
- **NumPy-style docstrings**: imperative first sentence (fits an IDE tooltip), array **shapes**
  stated, **units** explicit (Hz, s, rad, µV), a "When to use" note for metrics, runnable examples,
  references with DOI.
- **One concept per notebook / one responsibility per file**; type hints on public signatures;
  `logging` (not `print`) in library code; random seeds fixed and documented.

If you help the learner *write* code, hold to these — they keep the course legible to the next
reader, human or AI.
