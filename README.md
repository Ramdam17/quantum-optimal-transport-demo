# Quantum Optimal Transport — Hands-On Course

> 🚧 **Under active restructuring (2026-05).** Moving from 16 dense notebooks to ~57 fine-grained,
> one-concept-per-notebook modules across four movements (quantum foundations → information &
> geometry → classical OT → quantum OT). The structure and docs will change; a full README lands
> when the restructure completes.

A hands-on course from classical optimal transport to its quantum generalization.

- **Teaser:** `docs/qot-course-teaser.md` *(being rewritten)*

## Setup

```bash
uv sync --extra dev
uv run pytest
```

All deliverables (code, notebooks, PDF summaries, glossary, dictionary) are in English.

## Using this course with an AI assistant

Most learners follow this course with an AI assistant (Claude, ChatGPT, Gemini) open alongside —
and that's encouraged. To get the most out of it:

- Point your assistant at **`AGENTS.md`** — it tells the assistant how to *tutor* you (guide and
  hint, not just hand over answers).
- Ask it to **explain the intuition** first, to **quiz you**, and to **check your "Your turn"**
  attempts *after* you've tried them yourself. The struggle is where the learning happens.
- Use **`llms.txt`** (or just ask) to find which notebook covers a concept and what it builds on.

Treat the assistant as a patient tutor, not an answer key — you'll learn far more.

## Credentials

Real IBM Quantum hardware is optional and simulator-first. Never hardcode an API
token. Run `QiskitRuntimeService.save_account(...)` once locally (or put the token
in a git-ignored `.env`); notebooks then call `QiskitRuntimeService()` with no token.
