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
