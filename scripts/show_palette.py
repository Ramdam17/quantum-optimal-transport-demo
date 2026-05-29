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
