"""Plain data model for an extracted DIALux evo study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Luminaire:
    id: str = ""
    name: str = ""
    manufacturer: str = ""
    article_no: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotation: float = 0.0          # degrees about Z
    tilt: float = 0.0
    luminous_flux: Optional[float] = None    # lm
    power: Optional[float] = None            # W
    efficacy: Optional[float] = None         # lm/W
    cct: Optional[float] = None              # K
    cri: Optional[float] = None
    count: int = 1
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        bits = [b for b in (self.manufacturer, self.name or self.article_no) if b]
        return " ".join(bits) or self.id or "Luminaire"


@dataclass
class Polygon:
    """A closed outline in plan view."""
    points: List[Tuple[float, float]] = field(default_factory=list)
    name: str = ""

    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def area(self) -> float:
        """Shoelace area."""
        p = self.points
        if len(p) < 3:
            return 0.0
        s = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0


@dataclass
class Room:
    id: str = ""
    name: str = ""
    outline: Optional[Polygon] = None
    height: Optional[float] = None
    reflectance_ceiling: Optional[float] = None
    reflectance_walls: Optional[float] = None
    reflectance_floor: Optional[float] = None
    maintenance_factor: Optional[float] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def area(self) -> float:
        return self.outline.area() if self.outline else 0.0


@dataclass
class CalcGrid:
    """A recovered illuminance grid for one calculation surface."""
    group: str = ""
    name: str = ""
    values: Optional[np.ndarray] = None      # 2-D, lux
    nx: int = 0
    ny: int = 0
    extent: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1
    height: Optional[float] = None
    source: str = ""                          # which file it came from
    confidence: float = 0.0
    other_candidates: List[dict] = field(default_factory=list, repr=False)
    # Other high-scoring float runs found in the same source file that were
    # NOT used for `values`. On evo files with genuinely separate direct /
    # indirect / total illuminance arrays, one of these may be exactly that
    # — but nothing here is confirmed against real DIALux output, so treat
    # each entry as a lead to inspect (see other_candidates entries' offset),
    # not as a labelled result.

    # -- the quantities DIALux prints on its result sheets -------------------

    def metrics(self) -> Dict[str, float]:
        if self.values is None or self.values.size == 0:
            return {}
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return {}
        e_avg = float(v.mean())
        e_min = float(v.min())
        e_max = float(v.max())
        return {
            "Eav": e_avg,
            "Emin": e_min,
            "Emax": e_max,
            "u0": (e_min / e_avg) if e_avg else 0.0,     # Emin / Eav
            "Emin/Emax": (e_min / e_max) if e_max else 0.0,
            "Emax/Eav": (e_max / e_avg) if e_avg else 0.0,
            "median": float(np.median(v)),
            "std": float(v.std()),
            "points": int(v.size),
        }


@dataclass
class EnvironmentReading:
    """
    Summary illuminance values recovered from Dataenvillum*.rsl — DIALux's
    per-surface (wall/ceiling/floor) reflected-light bookkeeping, as opposed
    to the full working-plane grid. Too few numbers to be a raster; treated
    as labelled scalars instead.
    """
    group: str = ""
    values: List[float] = field(default_factory=list)
    source: str = ""



@dataclass
class Study:
    """Everything recovered from one .evo archive."""
    project_name: str = ""
    description: str = ""
    author: str = ""
    company: str = ""
    dialux_version: str = ""
    created: str = ""
    rooms: List[Room] = field(default_factory=list)
    luminaires: List[Luminaire] = field(default_factory=list)
    grids: List[CalcGrid] = field(default_factory=list)
    environment_readings: List["EnvironmentReading"] = field(default_factory=list)
    geometry: List[Polygon] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_path: str = ""

    # -- aggregate figures ---------------------------------------------------

    def total_power(self) -> Optional[float]:
        vals = [l.power * l.count for l in self.luminaires if l.power]
        return sum(vals) if vals else None

    def total_flux(self) -> Optional[float]:
        vals = [l.luminous_flux * l.count for l in self.luminaires
                if l.luminous_flux]
        return sum(vals) if vals else None

    def total_area(self) -> float:
        a = sum(r.area for r in self.rooms)
        if a:
            return a
        for p in self.geometry:
            a = max(a, p.area())
        return a

    def power_density(self) -> Optional[float]:
        """W/m2 — DIALux prints this as 'Specific connected load'."""
        p, a = self.total_power(), self.total_area()
        return (p / a) if (p and a) else None

    def power_density_per_100lx(self) -> Optional[float]:
        pd = self.power_density()
        if not pd or not self.grids:
            return None
        m = self.grids[0].metrics()
        eav = m.get("Eav")
        return (pd / eav * 100.0) if eav else None

    def luminaire_schedule(self) -> List[dict]:
        """Group identical luminaires into a bill-of-quantities table."""
        buckets: Dict[str, dict] = {}
        for l in self.luminaires:
            key = f"{l.manufacturer}|{l.name}|{l.article_no}"
            b = buckets.setdefault(key, {
                "manufacturer": l.manufacturer,
                "name": l.name or l.article_no or "Luminaire",
                "article_no": l.article_no,
                "flux": l.luminous_flux,
                "power": l.power,
                "cct": l.cct,
                "count": 0,
            })
            b["count"] += l.count
        rows = list(buckets.values())
        for r in rows:
            r["total_power"] = (r["power"] * r["count"]) if r["power"] else None
            r["total_flux"] = (r["flux"] * r["count"]) if r["flux"] else None
        rows.sort(key=lambda r: -r["count"])
        return rows

    def bounds(self) -> Tuple[float, float, float, float]:
        """Plan-view bounding box across rooms, geometry and luminaires."""
        xs, ys = [], []
        for r in self.rooms:
            if r.outline and r.outline.points:
                x0, y0, x1, y1 = r.outline.bounds()
                xs += [x0, x1]; ys += [y0, y1]
        for p in self.geometry:
            if p.points:
                x0, y0, x1, y1 = p.bounds()
                xs += [x0, x1]; ys += [y0, y1]
        for l in self.luminaires:
            xs.append(l.x); ys.append(l.y)
        for g in self.grids:
            if g.extent:
                xs += [g.extent[0], g.extent[2]]
                ys += [g.extent[1], g.extent[3]]
        if not xs:
            return (0.0, 0.0, 10.0, 10.0)
        pad = max((max(xs) - min(xs)), (max(ys) - min(ys))) * 0.08 + 0.5
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def aspect(self) -> float:
        x0, y0, x1, y1 = self.bounds()
        h = (y1 - y0) or 1.0
        return (x1 - x0) / h
