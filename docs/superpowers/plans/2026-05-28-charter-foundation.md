# Charter Foundation Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the centralized graphic charter (palette + style + notebook template) that every QOT notebook will depend on — before any notebook is written or moved.

**Architecture:** A new single-source-of-truth palette module `qot_course/colors.py` holds the soft, warm pastel palette (a family resemblance with Rémy's PPSP connectivity course). `viz.py` sources its colors and colormaps from it, keeping `SOURCE_COLOR`/`TARGET_COLOR`/`FLOW_COLOR` as backward-compatible aliases so the 16 legacy notebooks keep running until they are migrated. A swatch script lets Rémy validate the palette visually; a guard script forbids hardcoded hex in new notebooks; a template doc encodes the charter (sections + voice) for every future notebook.

**Tech Stack:** Python 3.12, numpy, matplotlib, pytest. Style via `mpl.rcParams`. Palette as a plain dict + `matplotlib.colors.LinearSegmentedColormap`.

**Charter rules enforced here (from the design spec §6):** centralized palette, no hardcoded hex in notebooks, fixed fig dims/fonts, warm/empowering voice, no decorative emojis.

---

## File structure

- Create `src/qot_course/colors.py` — the palette (`COLORS` dict, semantic colormaps, helpers). Single source of truth.
- Modify `src/qot_course/viz.py` — source colors/colormaps from `colors.py`; keep legacy aliases.
- Create `tests/test_colors.py` — palette validity + viz-alias consistency + style application.
- Create `scripts/show_palette.py` — render swatches for visual validation.
- Create `scripts/check_no_hardcoded_hex.py` — guard: no `#rrggbb` literals in new-style notebooks.
- Create `docs/notebook_template.md` — the charter notebook skeleton + voice guidance.

---

## Proposed QOT palette (for Rémy's visual validation)

Soft, warm pastels — same family as the PPSP course, mapped to QOT roles. Concrete hex below;
validate visually via `scripts/show_palette.py` (Task 4) and the first re-rendered figures.

| Key | Hex | Role |
|-----|-----|------|
| `source` | `#9B8FD4` | soft periwinkle — the pile we *have* |
| `target` | `#E8B864` | warm amber — the pile we *want* |
| `flow` | `#88C9A1` | soft sage — mass in motion (arrows/flow) |
| `quantum` | `#7EB8DA` | sky blue — quantum objects (states, ρ) |
| `highlight` | `#F4A4B8` | rose — emphasis / the punchline |
| `negative` | `#E17055` | coral — negative values / Re<0 |
| `zero` | `#FFFFFF` | white — zero midpoint |
| `positive` | `#5BB8B0` | soft teal — positive values / Re>0 |
| `grid` | `#E2E2EC` | gridlines |
| `text` | `#2C3E50` | text / strong ink |
| `muted` | `#9AA0AA` | secondary annotation |
| `background` | `#FFFFFF` | figure background |

Colormaps derived from these: `CMAP_PLAN` (white→flow, transport plans), `CMAP_COST`
(white→source, cost matrices), `CMAP_DENSITY` (coral↔white↔teal, density-matrix Re/Im).

---

## Task 1: The palette module

**Files:**
- Create: `src/qot_course/colors.py`
- Test: `tests/test_colors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colors.py
import re
import matplotlib.colors as mcolors
from qot_course import colors

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
REQUIRED = {
    "source", "target", "flow", "quantum", "highlight",
    "negative", "zero", "positive", "grid", "text", "muted", "background",
}


def test_required_palette_keys_present():
    assert REQUIRED <= set(colors.COLORS)


def test_all_palette_values_are_valid_hex():
    for key, value in colors.COLORS.items():
        assert HEX.match(value), f"{key}={value!r} is not #rrggbb"


def test_colormaps_are_matplotlib_colormaps():
    for cmap in (colors.CMAP_PLAN, colors.CMAP_COST, colors.CMAP_DENSITY):
        assert isinstance(cmap, mcolors.Colormap)


def test_density_colormap_is_diverging_white_centre():
    mid = colors.CMAP_DENSITY(0.5)
    assert mid[0] > 0.95 and mid[1] > 0.95 and mid[2] > 0.95  # white midpoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qot_course.colors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qot_course/colors.py
"""Single source of truth for the course's graphic charter palette.

Soft warm pastels, a family resemblance with the PPSP connectivity course.
Never hardcode hex in notebooks or modules — import from here.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

COLORS: dict[str, str] = {
    # Optimal-transport roles
    "source": "#9B8FD4",      # soft periwinkle — the pile we have
    "target": "#E8B864",      # warm amber — the pile we want
    "flow": "#88C9A1",        # soft sage — mass in motion
    # Quantum / accents
    "quantum": "#7EB8DA",     # sky blue — quantum objects (states, rho)
    "highlight": "#F4A4B8",   # rose — emphasis / the punchline
    # Diverging (correlations, density-matrix Re/Im)
    "negative": "#E17055",    # coral
    "zero": "#FFFFFF",
    "positive": "#5BB8B0",    # soft teal
    # Neutrals
    "grid": "#E2E2EC",
    "text": "#2C3E50",
    "muted": "#9AA0AA",
    "background": "#FFFFFF",
}

CMAP_PLAN = LinearSegmentedColormap.from_list(
    "qot_plan", [COLORS["background"], COLORS["flow"]]
)
CMAP_COST = LinearSegmentedColormap.from_list(
    "qot_cost", [COLORS["background"], COLORS["source"]]
)
CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "qot_density", [COLORS["negative"], COLORS["zero"], COLORS["positive"]]
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/colors.py tests/test_colors.py
git commit -m "feat(charter): add central palette module qot_course.colors"
```

---

## Task 2: Route viz.py through the palette (with backward-compatible aliases)

**Files:**
- Modify: `src/qot_course/viz.py:15-20` (the color/colormap constants)
- Test: `tests/test_colors.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_colors.py)**

```python
def test_viz_aliases_point_at_palette():
    from qot_course import viz
    assert viz.SOURCE_COLOR == colors.COLORS["source"]
    assert viz.TARGET_COLOR == colors.COLORS["target"]
    assert viz.FLOW_COLOR == colors.COLORS["flow"]
    assert viz.CMAP_PLAN is colors.CMAP_PLAN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::test_viz_aliases_point_at_palette -v`
Expected: FAIL — `viz.SOURCE_COLOR == "#7c3aed"` ≠ palette value; `viz.CMAP_PLAN` is the string `"plasma"`.

- [ ] **Step 3: Edit viz.py — replace the literal constants (lines 15-20) with palette-sourced ones**

Replace:
```python
# Vibrant, consistent palette used across the whole course.
SOURCE_COLOR = "#7c3aed"  # violet — the distribution we have
TARGET_COLOR = "#f59e0b"  # amber  — the distribution we want
FLOW_COLOR = "#10b981"  # emerald — mass in motion
CMAP_COST = "magma"  # cost matrices
CMAP_PLAN = "plasma"  # transport plans (mass)
```
with:
```python
# Palette lives in qot_course.colors (single source of truth). These names are
# kept as backward-compatible aliases for the existing notebooks.
from qot_course.colors import COLORS, CMAP_COST, CMAP_PLAN, CMAP_DENSITY

SOURCE_COLOR = COLORS["source"]  # the distribution we have
TARGET_COLOR = COLORS["target"]  # the distribution we want
FLOW_COLOR = COLORS["flow"]      # mass in motion
```

- [ ] **Step 4: Run the full suite to verify nothing broke**

Run: `uv run pytest -q`
Expected: PASS (162 passed — the 161 baseline + the new alias test). The legacy notebooks’ viz calls still resolve because the aliases are preserved.

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/viz.py tests/test_colors.py
git commit -m "refactor(charter): source viz colors from qot_course.colors"
```

---

## Task 3: Charter rcParams in use_course_style()

**Files:**
- Modify: `src/qot_course/viz.py:22-47` (the `_STYLE` dict + `use_course_style`)
- Test: `tests/test_colors.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_use_course_style_applies_charter():
    import matplotlib as mpl
    from qot_course import viz
    viz.use_course_style()
    assert mpl.rcParams["grid.color"] == colors.COLORS["grid"]
    assert mpl.rcParams["text.color"] == colors.COLORS["text"]
    assert mpl.rcParams["figure.dpi"] == 110
    assert mpl.rcParams["axes.titlesize"] == 14  # charter title size
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::test_use_course_style_applies_charter -v`
Expected: FAIL — `grid.color` is `"#e2e8f0"` (not palette), `axes.titlesize` is 15, `text.color` unset.

- [ ] **Step 3: Edit viz.py `_STYLE` — route neutral colors through COLORS and set the charter sizes**

In the `_STYLE` dict change these entries:
```python
    "axes.edgecolor": COLORS["grid"],
    "grid.color": COLORS["grid"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlesize": 14,
```
(Leave the rest of `_STYLE` as-is. `COLORS` is already imported from Task 2.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (163 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qot_course/viz.py tests/test_colors.py
git commit -m "feat(charter): apply palette neutrals and charter sizes in use_course_style"
```

---

## Task 4: Palette swatch script (visual validation)

**Files:**
- Create: `scripts/show_palette.py`
- Test: `tests/test_colors.py` (append a smoke test)

- [ ] **Step 1: Write the failing smoke test**

```python
def test_show_palette_builds_a_figure():
    import matplotlib
    matplotlib.use("Agg")
    from scripts.show_palette import build_swatch_figure
    fig = build_swatch_figure()
    assert fig.axes  # at least one axis drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::test_show_palette_builds_a_figure -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.show_palette'`.

- [ ] **Step 3: Write the script**

```python
# scripts/show_palette.py
"""Render the charter palette as labelled swatches for visual validation.

Run:  uv run python scripts/show_palette.py   (writes figures/palette.png + shows)
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from qot_course.colors import COLORS


def build_swatch_figure() -> plt.Figure:
    items = list(COLORS.items())
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(items) + 1))
    for i, (name, hex_value) in enumerate(items):
        y = len(items) - 1 - i
        ax.add_patch(plt.Rectangle((0, y), 1, 0.9, color=hex_value))
        ax.text(1.1, y + 0.45, f"{name}  {hex_value}", va="center", fontsize=11)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(items))
    ax.axis("off")
    ax.set_title("QOT charter palette", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    fig = build_swatch_figure()
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "palette.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / 'palette.png'}")
    plt.show()
```

- [ ] **Step 4: Run test, then render for Rémy**

Run: `uv run pytest tests/test_colors.py::test_show_palette_builds_a_figure -v`  → PASS
Run: `uv run python scripts/show_palette.py`  → prints the PNG path; **Rémy validates the palette visually here.**

- [ ] **Step 5: Commit**

```bash
git add scripts/show_palette.py tests/test_colors.py
git commit -m "feat(charter): add palette swatch script for visual validation"
```

---

## Task 5: No-hardcoded-hex guard for new notebooks

**Files:**
- Create: `scripts/check_no_hardcoded_hex.py`
- Test: `tests/test_colors.py` (append)

Scope: scans **new-style** notebooks only — files matching `notebooks/[0-9][0-9]_*/[0-9][0-9]_*.ipynb`.
Legacy `notebooks/s*.ipynb` are exempt (they will be migrated in later plans). With no new
notebooks yet, the check passes trivially and guards every notebook we add from here on.

- [ ] **Step 1: Write the failing test**

```python
def test_no_hardcoded_hex_checker_runs_clean_on_new_notebooks():
    from scripts.check_no_hardcoded_hex import find_hardcoded_hex
    offenders = find_hardcoded_hex()
    assert offenders == [], f"hardcoded hex found: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::test_no_hardcoded_hex_checker_runs_clean_on_new_notebooks -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_no_hardcoded_hex'`.

- [ ] **Step 3: Write the checker**

```python
# scripts/check_no_hardcoded_hex.py
"""Guard: forbid hardcoded #rrggbb literals in new-style notebooks.

Colours must come from qot_course.colors. Legacy notebooks/s*.ipynb are exempt
(migrated later). Run:  uv run python scripts/check_no_hardcoded_hex.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
ROOT = Path(__file__).resolve().parents[1]
NEW_NOTEBOOK_GLOB = "notebooks/[0-9][0-9]_*/[0-9][0-9]_*.ipynb"


def find_hardcoded_hex() -> list[str]:
    offenders: list[str] = []
    for nb_path in sorted(ROOT.glob(NEW_NOTEBOOK_GLOB)):
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for line in cell.get("source", []):
                if HEX.search(line):
                    offenders.append(f"{nb_path.relative_to(ROOT)}: {line.strip()}")
    return offenders


if __name__ == "__main__":
    found = find_hardcoded_hex()
    if found:
        print("Hardcoded hex (use qot_course.colors instead):")
        for line in found:
            print("  " + line)
        raise SystemExit(1)
    print("OK — no hardcoded hex in new notebooks.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py::test_no_hardcoded_hex_checker_runs_clean_on_new_notebooks -v`
Expected: PASS (no new notebooks yet → no offenders).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_no_hardcoded_hex.py tests/test_colors.py
git commit -m "feat(charter): add no-hardcoded-hex guard for new notebooks"
```

---

## Task 6: The notebook charter template

**Files:**
- Create: `docs/notebook_template.md`

This is prose, not code — it encodes the per-notebook charter (sections + voice) every future
notebook follows. It is the scaffold the module plans (C–G) instantiate.

- [ ] **Step 1: Write the template**

````markdown
# docs/notebook_template.md
# Notebook charter template (QOT course)

Every notebook is ONE concept (a "brick") or a "synthesis". Plain `.ipynb`, output-free in git.
Voice: warm, empowering, celebratory — AND rigorous. No decorative emojis. No "obviously /
simply / trivially / just". Celebrate what the learner just achieved; frame difficulty as growth.

## Cell order
1. **Header** (markdown): `# NN — Title`; one-line purpose; **Prerequisites** (notebook ids);
   **What you'll be able to do** (3-5 action-verb objectives). A warm one-line welcome is encouraged.
2. **Imports** (code): stdlib / third-party / `qot_course` local. `np.random.seed(...)`;
   `from qot_course import viz; viz.use_course_style()`. **Never** hardcode hex — use `qot_course.colors`.
3. **Body sections** (one concept), each: *intuition* (markdown) → *implementation* (code) →
   **"Read the figure / output"** (markdown — always explain what we see, kindly and concretely).
4. **Your turn** (markdown): 2-3 small exercises, tiered easy→harder (no emoji labels — words).
5. **Summary** (markdown): "What you built" bullets that *celebrate the accomplishment* + dictionary row.
6. **References** (markdown): cited papers; `Previous:` / `Next:` links within the module.

## Synthesis notebooks
Assume their bricks; integrate, deliver the punchline + dictionary row + (where relevant) the
hardware/application demo. Lighter than a from-scratch notebook.

## Figures
Use `qot_course.viz` helpers and `qot_course.colors`. Fixed dims/DPI from the charter. Every
figure is followed by a "Read the figure" paragraph.
````

- [ ] **Step 2: Commit**

```bash
git add docs/notebook_template.md
git commit -m "docs(charter): add the notebook charter + voice template"
```

---

## Task 7: Mark the charter done in STATE.md

**Files:**
- Modify: `docs/superpowers/STATE.md`

- [ ] **Step 1: Update STATE.md**

- Tick `[x] Charter finalized (palette module + use_course_style)` under "Not started — build".
- Under "Open questions", change the palette line to: `- Palette: proposed in Plan A, pending Rémy's visual sign-off via scripts/show_palette.py.`
- Update `**Last updated:**` and `## Next action` to point at Plan B (folder reorg).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/STATE.md
git commit -m "docs(restructure): mark charter foundation complete in STATE"
```

---

## Self-review (done at authoring time)

- **Spec coverage:** implements spec §6 charter (centralized palette ✓, no hardcoded hex ✓, fixed
  dims/fonts ✓, voice/no-emoji encoded in the template ✓). The exact-palette open item is resolved
  by Task 1 + visual validation in Task 4.
- **Placeholders:** none — every code step has complete code; every command has expected output.
- **Type consistency:** `COLORS`, `CMAP_PLAN/COST/DENSITY` names are identical across Tasks 1-4;
  `find_hardcoded_hex()` and `build_swatch_figure()` names match their tests.
- **Note:** the legacy 16 notebooks keep their *vibrant* look only via the alias values until they
  are migrated; after Task 2 the aliases carry the *new* palette, so re-running a legacy notebook
  already shows the new colours (intended — no regression, just the new charter).
