# Quantum Optimal Transport — Hands-On Course

A 16-session hands-on course from classical optimal transport to quantum optimal transport.

- **Teaser (share this):** `docs/qot-course-teaser.md`
- **Design spec:** `docs/superpowers/specs/2026-05-27-qot-course-design.md`
- **Plans:** `docs/superpowers/plans/`

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
