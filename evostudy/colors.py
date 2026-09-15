"""
Colour scales that match the look of DIALux evo output sheets.

DIALux uses a banded (discrete) false-colour scale for illuminance rather
than a continuous gradient, with the bands chosen from a "nice numbers"
sequence based on the maximum value in the calculation.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

# Band colours, dark (low lux) -> light (high lux). Approximates the evo
# false-colour rendering.
FALSE_COLOUR_BANDS: List[str] = [
    "#1b1464",  # deep indigo
    "#2b3a9c",
    "#2f6fd0",
    "#2bb1d6",
    "#2ecfa2",
    "#7ad64f",
    "#c9e04a",
    "#f5e04a",
    "#f7b93e",
    "#f28a2e",
    "#e85c2a",
    "#d63333",
    "#ef6f8f",
    "#f8b6c8",
    "#fde9f0",   # pale tint, not pure white, so peaks stay visible
]

FALSE_COLOUR_CMAP = ListedColormap(FALSE_COLOUR_BANDS, name="dialux_false")

# Continuous variant for smooth renders
FALSE_COLOUR_SMOOTH = LinearSegmentedColormap.from_list(
    "dialux_false_smooth", FALSE_COLOUR_BANDS, N=512
)

# Greyscale plan styling, matching evo's technical drawings
PLAN = {
    "wall": "#1a1a1a",
    "wall_lw": 1.8,
    "room_fill": "#fbfbfa",
    "grid": "#d8d8d4",
    "calc_area": "#5b8def",
    "luminaire": "#f5a623",
    "luminaire_edge": "#7a4d00",
    "dimension": "#555555",
    "text": "#1a1a1a",
    "accent": "#c8102e",
}

# Standard EN 12464-1 maintained illuminance steps, used for band selection
EN_SERIES = [
    0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300,
    500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000,
    30000, 50000, 75000, 100000,
]


def nice_levels(vmin: float, vmax: float, n: int = 10) -> List[float]:
    """
    Choose contour/band levels the way a lighting report would: rounded
    values from the EN series, spanning the data.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return [0.0, 1.0]

    inside = [v for v in EN_SERIES if vmin <= v <= vmax]
    if len(inside) >= 4:
        if len(inside) > n:
            step = max(1, len(inside) // n)
            inside = inside[::step]
        levels = sorted(set([round(vmin, 2)] + inside + [round(vmax, 2)]))
        # A handful of EN steps can leave the scale very coarse. Subdivide the
        # widest gaps until the band count is useful for reading a plan.
        while len(levels) < max(6, n - 2):
            gaps = [(levels[i + 1] - levels[i], i) for i in range(len(levels) - 1)]
            widest, i = max(gaps)
            if widest <= 0:
                break
            levels.insert(i + 1, round(levels[i] + widest / 2.0, 2))
            levels = sorted(set(levels))
        return levels

    # Fall back to linear rounding
    span = vmax - vmin
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mag * mult:
            step = mag * mult
            break
    else:
        step = mag * 10
    start = np.floor(vmin / step) * step
    levels = list(np.arange(start, vmax + step, step))
    return [float(v) for v in levels] or [vmin, vmax]


def band_norm(levels: Sequence[float]) -> BoundaryNorm:
    """Discrete normaliser matching the number of bands to the palette."""
    n = max(len(levels) - 1, 1)
    cmap = false_colour_cmap(n)
    return BoundaryNorm(list(levels), cmap.N)


def false_colour_cmap(n_bands: int) -> ListedColormap:
    """Resample the band palette to exactly n_bands colours."""
    n_bands = max(1, int(n_bands))
    idx = np.linspace(0, len(FALSE_COLOUR_BANDS) - 1, n_bands)
    cols = [FALSE_COLOUR_BANDS[int(round(i))] for i in idx]
    return ListedColormap(cols)


def value_text_colour(value: float, vmin: float, vmax: float) -> str:
    """Black or white label text, whichever stays readable on the band."""
    if vmax <= vmin:
        return "#000000"
    t = (value - vmin) / (vmax - vmin)
    return "#ffffff" if t < 0.45 else "#111111"
