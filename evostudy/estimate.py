"""
Per-fixture illuminance contribution — ESTIMATED, not extracted.

DIALux's exported .rsl results are the combined total for the whole room;
evo does not normally save a separate grid per luminaire. There is no file
in the archive to recover a true per-fixture map from.

What this module does instead: compute each luminaire's *direct* component
on the same grid, from its recovered position and luminous flux, using the
standard point-source photometric model:

    E = (I0 * cos^3(theta)) / h^2          (Lambertian, no IES data)
    I0 = flux / pi                          (nadir intensity of a Lambertian
                                              source of the given flux)
    h  = mounting height - working-plane height
    theta = angle from the luminaire's nadir to the point

This ignores the luminaire's real photometric distribution (its .ies/.ldt
file is not read), inter-reflection, and shadowing from geometry — it is a
first-order approximation useful for seeing each fixture's footprint and
relative contribution, not a substitute for DIALux's own calculation.

Every value this module produces is tagged estimated=True end to end so it
can never be confused with a recovered result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .model import CalcGrid, Luminaire, Study


@dataclass
class FixtureContribution:
    luminaire: Luminaire
    label: str
    values: np.ndarray          # 2-D, lux, same shape as the parent grid
    extent: Tuple[float, float, float, float]
    estimated: bool = True

    def metrics(self) -> dict:
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return {}
        return {
            "Eav": float(v.mean()),
            "Emin": float(v.min()),
            "Emax": float(v.max()),
            "share_of_peak": float(v.max()),
        }


def direct_component(luminaire: Luminaire, xs: np.ndarray, ys: np.ndarray,
                     plane_height: float) -> np.ndarray:
    """
    Evaluate the Lambertian point-source model over a grid of (xs, ys)
    positions at the given working-plane height. Returns a (len(ys), len(xs))
    array in lux.
    """
    X, Y = np.meshgrid(xs, ys)
    h = max(luminaire.z - plane_height, 0.05)   # avoid a divide-by-zero
    flux = luminaire.luminous_flux or 0.0
    if flux <= 0:
        return np.zeros_like(X)

    I0 = flux / np.pi
    dx = X - luminaire.x
    dy = Y - luminaire.y
    d2 = dx * dx + dy * dy
    r2 = d2 + h * h
    cos_t = h / np.sqrt(r2)
    return I0 * (cos_t ** 3) / r2


def estimate_all_fixtures(study: Study, grid: CalcGrid,
                          max_fixtures: int = 60
                          ) -> List[FixtureContribution]:
    """
    Build one FixtureContribution per luminaire in the study, on the same
    grid resolution and extent as `grid`. Skips silently past `max_fixtures`
    to keep interactive output usable — that limit only affects how many
    individually-selectable maps are produced, not the totals shown
    elsewhere, which always cover every luminaire.
    """
    if grid.extent is None or grid.nx <= 0 or grid.ny <= 0:
        return []
    x0, y0, x1, y1 = grid.extent
    xs = np.linspace(x0, x1, grid.nx)
    ys = np.linspace(y0, y1, grid.ny)
    plane_h = grid.height if grid.height is not None else 0.8

    out = []
    for lum in study.luminaires[:max_fixtures]:
        vals = direct_component(lum, xs, ys, plane_h)
        out.append(FixtureContribution(
            luminaire=lum, label=lum.label, values=vals,
            extent=grid.extent, estimated=True))
    return out


def combined_direct_estimate(contributions: List[FixtureContribution]
                             ) -> Optional[np.ndarray]:
    """Sum of every fixture's estimated direct component — a sanity check
    against the recovered total (it will always read lower, since it omits
    interreflection)."""
    if not contributions:
        return None
    total = np.zeros_like(contributions[0].values)
    for c in contributions:
        total += c.values
    return total
