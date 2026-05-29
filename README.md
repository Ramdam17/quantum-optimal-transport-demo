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

## Credentials

Real IBM Quantum hardware is optional and simulator-first. Never hardcode an API
token. Run `QiskitRuntimeService.save_account(...)` once locally (or put the token
in a git-ignored `.env`); notebooks then call `QiskitRuntimeService()` with no token.
