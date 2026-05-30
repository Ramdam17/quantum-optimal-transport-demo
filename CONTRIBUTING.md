# Contributing

Thank you for your interest in improving this course. Whether you are fixing a typo, sharpening an
explanation, adding an exercise, or extending the library, your contribution is welcome — and this
guide will help you make it cleanly.

## Development setup

```bash
uv sync --extra dev
```

## Running the checks

Before opening a pull request, make sure everything is green:

```bash
uv run pytest                      # full suite (src + offline notebook execution)
uv run pytest -m "not notebooks"   # fast inner loop (skips the slow notebook-execution tests)
uv run ruff check .                # lint
uv run ruff format --check         # format check
uv run python scripts/check_no_hardcoded_hex.py   # palette guard (no hardcoded colours)
```

The notebook-execution tests run every notebook end-to-end on an offline simulator — no network and no
IBM credentials required, so the committed notebooks stay output-free.

## Standards

The course holds a consistent standard so it reads as a coherent whole:

- **One concept per notebook.** Short focused "bricks" build to a "synthesis"; never cram several new
  ideas into one notebook.
- **Notebooks are output-free in git.** Clear outputs before committing.
- **Centralized palette, no hardcoded colours.** Use `qot_course.viz` / `qot_course.colors`; never put a
  hex literal in a notebook (the palette guard enforces this).
- **NumPy-style docstrings** with array shapes and units, a "when to use" note, a runnable example, and
  source citations (with DOI).
- **Cite sources; flag heuristic vs. established results.** Validate numbers against closed forms where
  one exists.
- **Voice = warm, empowering, and rigorous.** Celebrate what the learner achieves; frame difficulty as
  growth, never a wall. Avoid the dismissive qualifiers "obviously / simply / trivially / just" and any
  condescension; no decorative emoji. Warmth lives in the words, and never softens the science.

Every figure is followed by a "Read the figure" explanation; every notebook ends with tiered "Your turn"
exercises, a "What you built" recap, and a dictionary row where relevant. See `AGENTS.md` for the
tutoring philosophy and `docs/notebook_template.md` for the full notebook contract.

## Proposing changes

- Work on a **feature branch** and open a **pull request to `main`** (please do not commit directly to
  `main`).
- Keep `uv run pytest` green; commit per notebook / per logical change, with clear messages.
- If you change notebooks, regenerate the navigation index: `uv run python scripts/gen_llms_txt.py`.

## License of contributions

By contributing, you agree that your contributions are licensed under the repository's terms — the **MIT
License** for code (`src/`, `scripts/`, `tests/`) and **CC BY 4.0** for course content (`notebooks/`,
`docs/`). See [`LICENSE`](LICENSE) and [`LICENSE-CONTENT`](LICENSE-CONTENT).
