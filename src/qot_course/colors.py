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
