"""
Recovery of illuminance results from DIALux evo's .rsl files.

Layout inside Project/Results/<Group_xx>/ :

    Dataillumqt0.rsl        illumination quantities  <- the numbers we want
    Dataillumqt0Toc.rsl     its index / record table
    Dataclptr0.rsl          calculation point geometry (the grid itself)
    Dataclptr0Toc.rsl
    Dataenvillum0.rsl       environment (wall/ceiling/floor) illumination
    Datagenericids0.rsl     id lookup table
    Datagenericstrs0.rsl    string lookup table (surface / group names)
    Datavislightemsurf0.rsl visible light emitting surfaces

The format is proprietary and undocumented. This module recovers the value
arrays statistically (see binfmt.py) rather than by parsing a spec, then
reshapes them into a 2-D grid.

ALWAYS cross-check the recovered Eav/Emin/Emax against DIALux's own output
sheet the first time you use this on a new project. Once you confirm the
offsets line up for your evo version, they will stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .archive import Entry
from .binfmt import (
    LUX_RANGE,
    METRE_RANGE,
    FloatRun,
    find_ascii_strings,
    find_float_runs,
    infer_grid_shape,
    score_as_coordinates,
    score_as_illuminance,
)
from .model import CalcGrid


@dataclass
class RslCandidate:
    run: FloatRun
    score: float
    kind: str      # 'illuminance' | 'coordinates'


def analyse_rsl(data: bytes, kind: str = "illuminance") -> List[RslCandidate]:
    """Score every float run in one .rsl file."""
    if kind == "coordinates":
        runs = find_float_runs(data, value_range=METRE_RANGE, min_count=24)
        scored = [RslCandidate(r, score_as_coordinates(r), "coordinates")
                  for r in runs]
    else:
        runs = find_float_runs(data, value_range=LUX_RANGE, min_count=48)
        scored = [RslCandidate(r, score_as_illuminance(r), "illuminance")
                  for r in runs]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def _planarity(pts: np.ndarray) -> float:
    """
    0..1 score for 'these points lie on one plane parallel to an axis'.
    A DIALux working-plane grid has a constant Z, so one column has almost
    no spread. Higher is better.
    """
    if pts.shape[0] < 4:
        return 0.0
    spreads = np.ptp(pts, axis=0)
    biggest = spreads.max()
    if biggest <= 0:
        return 0.0
    return float(1.0 - (spreads.min() / biggest))


def extract_point_cloud(data: bytes, expect: Optional[int] = None
                        ) -> Optional[np.ndarray]:
    """
    Pull an (N,3) array of calculation point coordinates out of
    Dataclptr*.rsl.

    The float scanner often picks up a few header words that happen to decode
    as plausible coordinates, which would shift every subsequent triple by one
    and destroy the raster. So we test each possible start offset (0,1,2) and
    also trimming from the front, keeping whichever reshape is most planar —
    the real grid is planar, a misaligned one is not.
    """
    cands = analyse_rsl(data, kind="coordinates")
    best, best_score = None, -1.0

    for c in cands:
        v = c.run.values
        if v.size < 30:
            continue
        # Candidate start offsets: raw alignment, plus a front trim that makes
        # the length match an expected point count.
        offsets = {0, 1, 2}
        if expect:
            for pad in range(0, 8):
                off = v.size - (expect * 3) - pad
                if 0 <= off < v.size:
                    offsets.add(off)
        for off in sorted(offsets):
            n = (v.size - off) // 3 * 3
            if n < 30:
                continue
            pts = v[off:off + n].reshape(-1, 3)
            score = _planarity(pts) * 0.7 + c.score * 0.3
            if expect and pts.shape[0] == expect:
                score += 0.5
            if score > best_score:
                best, best_score = pts, score

    return best if best_score > 0.3 else None


def _nan_fill(grid: np.ndarray) -> np.ndarray:
    """Simple iterative neighbour fill, no SciPy needed."""
    g = grid.copy()
    for _ in range(8):
        nan = np.isnan(g)
        if not nan.any():
            break
        padded = np.pad(g, 1, mode="edge")
        stack = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
        ])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
        g[nan] = mean[nan]
    g[np.isnan(g)] = np.nanmin(g) if np.isfinite(g).any() else 0.0
    return g


def extract_scalars(data: bytes, max_vals: int = 64) -> List[float]:
    """
    Pull a short list of plausible scalar readings out of a small .rsl file
    (Dataenvillum*, Datagenericids*, etc.) — these are too small to be a
    calculation grid and instead hold a handful of summary values, such as
    the mean illuminance DIALux reports for each room surface (wall/ceiling/
    floor), used in reflected-light bookkeeping.
    """
    cands = find_float_runs(data, value_range=LUX_RANGE, min_count=3)
    if not cands:
        return []
    best = max(cands, key=lambda r: r.count)
    v = best.values[:max_vals]
    return [round(float(x), 2) for x in v if np.isfinite(x)]



MAX_GRIDS_PER_GROUP = 6
MULTI_GRID_SCORE_FLOOR = 0.55
# A .rsl illuminance file routinely holds MORE than one quantity array side
# by side (observed on real projects: several float runs of comparable size
# but different zero-fraction / mean / max in the same file — consistent
# with distinct calculation surfaces or distinct quantities such as direct-
# only vs. a component that never truly reaches zero). Historically this
# module kept only the single top-scoring run and threw the rest away as
# `other_candidates` metadata that nothing ever rendered. It now rasterises
# every plausible run (above MULTI_GRID_SCORE_FLOOR, capped at
# MAX_GRIDS_PER_GROUP) into its own CalcGrid, so they all become separately
# selectable/viewable/exportable. What each one PHYSICALLY represents
# (direct? indirect? a second surface?) is still not confirmed against real
# DIALux output — see `_fingerprint_label` — so treat the label as a
# starting guess to check, not a certainty.


def _fingerprint_label(rank: int, s: dict, total_candidates: int) -> str:
    """
    A short, honest, checkable guess at what a candidate quantity array is,
    from its statistical shape alone. Zero exact-zero readings on a lit
    plane is unusual for anything except a component that is genuinely
    absent at some points (e.g. direct light in a shadowed area) or a
    component that is always present by construction (ambient/indirect,
    or a total that sums the two). This is a lead to check against
    DIALux's own printed Eav for the same surface, not a confirmed answer.
    """
    zero_frac = s.get("zeros", 0) / s["n"] if s.get("n") else 0.0
    if zero_frac > 0.15:
        pattern = (f"possible DIRECT-only component — {zero_frac:.0%} of "
                  f"points read exactly 0 (consistent with shadowed/unlit "
                  f"points)")
    elif zero_frac < 0.01:
        pattern = ("possible INDIRECT/total component — almost no exact-"
                  "zero points (consistent with an ambient or summed "
                  "quantity)")
    else:
        pattern = "unlabelled secondary quantity"
    tag = " [highest confidence score]" if rank == 0 else ""
    return f"candidate {rank + 1}/{total_candidates}: {pattern}{tag}"


def _build_grid(cand: RslCandidate, rank: int, total_candidates: int,
                group_name: str, base_name: str, illum_path: str,
                clptr: Optional[Entry],
                room_aspect: Optional[float],
                room_extent: Optional[Tuple[float, float, float, float]],
                other_candidates: List[dict]) -> Optional[CalcGrid]:
    """Rasterise ONE scored float run into a CalcGrid (route 1 then route 2)."""
    values = cand.run.values
    s = cand.run.stats()
    if not s:
        return None

    label = _fingerprint_label(rank, s, total_candidates)
    name = f"{base_name} — {label} (Eav≈{s['mean']:.0f} lx, n={s['n']})"

    grid = CalcGrid(group=group_name, name=name, source=illum_path,
                    confidence=cand.score, other_candidates=other_candidates)

    # --- route 1: real coordinates ----------------------------------------
    pts = None
    if clptr is not None:
        pts = extract_point_cloud(clptr.read(), expect=values.size)
        if pts is None:
            pts = extract_point_cloud(clptr.read())

    if pts is not None:
        res, pts_a = _best_alignment(pts, values, room_extent)
        if res is not None:
            arr, extent = res
            grid.values = _nan_fill(arr)
            grid.ny, grid.nx = arr.shape
            grid.extent = extent
            grid.height = float(
                np.median(pts_a[:, int(np.argmin(np.ptp(pts_a, axis=0)))]))
            grid.confidence = min(1.0, cand.score + 0.15)
            return grid

    # --- route 2: factorise ------------------------------------------------
    MAX_USABLE_ASPECT = 12.0  # a "grid" thinner than this per side is a
                               # degenerate sliver, not a viewable raster
    shape = infer_grid_shape(values.size, room_aspect)

    def _aspect(shp):
        a, b = shp
        return max(a, b) / max(min(a, b), 1)

    if shape is None or _aspect(shape) > MAX_USABLE_ASPECT:
        # Either no factorisation exists at all, or the only ones available
        # are unusably thin (e.g. n = 3^2 x 37201, a large prime — the best
        # near-square pair is (9, 37201)). Search downward for the nearest
        # trimmed length that factorises into something closer to square,
        # bounded so we never throw away more than ~2% of the points.
        n0 = values.size
        best_n, best_shape = None, shape
        limit = max(16, int(n0 * 0.02))
        for n in range(n0, n0 - limit, -1):
            if n <= 16:
                break
            cand_shape = infer_grid_shape(n, room_aspect)
            if cand_shape is not None and (
                    best_shape is None or _aspect(cand_shape) < _aspect(best_shape)):
                best_n, best_shape = n, cand_shape
                if _aspect(cand_shape) <= MAX_USABLE_ASPECT:
                    break
        if best_shape is not None:
            shape = best_shape
            values = values[:best_n]
    if shape is None:
        return None

    ny, nx = shape
    grid.values = values[:ny * nx].reshape(ny, nx)
    grid.ny, grid.nx = ny, nx
    grid.extent = room_extent or (0.0, 0.0, float(nx), float(ny))

    # A handful of stray padding bytes at the very start/end of the byte
    # scan can decode as exact 0.0 and survive the reshape sitting at the
    # array's first/last cells. On a component whose readings are almost
    # never truly zero (see `_fingerprint_label`'s zero_frac test) that
    # single spurious 0.0 corrupts Emin — and therefore u0 = Emin/Eav —
    # to exactly 0 even though the recovered map is otherwise smooth and
    # fully lit. Genuinely dark/shadowed components (large zero_frac, e.g.
    # a direct-only quantity) are left completely untouched: this only
    # fires when zeros are a tiny, isolated fraction of the array.
    zero_frac = s.get("zeros", 0) / s["n"] if s.get("n") else 0.0
    n_zero_now = int((grid.values == 0).sum())
    if 0 < n_zero_now <= 12 and zero_frac < 0.005:
        mask = grid.values == 0
        grid.values = grid.values.astype(float)
        grid.values[mask] = np.nan
        grid.values = _nan_fill(grid.values)
        grid.name += f" [{n_zero_now} isolated zero-cell(s) treated as scan padding and interpolated]"

    return grid


def read_group(group_name: str, entries: List[Entry],
               room_aspect: Optional[float] = None,
               room_extent: Optional[Tuple[float, float, float, float]] = None
               ) -> List[CalcGrid]:
    """
    Build every plausible CalcGrid from the .rsl files of a single result
    group — not just the single best-scoring one.

    Strategy per candidate, best first:
      1. Illuminance values + real point coordinates -> exact raster.
      2. Illuminance values only -> factorise N into the grid shape whose
         aspect ratio best matches the room, and stretch over the room extent.

    Returns a list ordered by confidence, highest first (empty if nothing
    usable was found). `grids[0]` is always the single highest-confidence
    quantity — safe to use alone wherever only "the" grid used to be used.
    """
    by_name = {e.name.lower(): e for e in entries}

    illum = next((e for k, e in by_name.items()
                  if "illumqt" in k and not k.endswith("toc.rsl")), None)
    clptr = next((e for k, e in by_name.items()
                  if "clptr" in k and not k.endswith("toc.rsl")), None)
    strs = next((e for k, e in by_name.items() if "genericstrs" in k), None)

    if illum is None:
        return []

    cands = analyse_rsl(illum.read(), kind="illuminance")
    if not cands:
        return []

    name = group_name
    if strs is not None:
        found = find_ascii_strings(strs.read(), min_len=3, limit=5)
        if found:
            name = found[0][1].replace("  [utf-16]", "").strip() or group_name

    # Every candidate is recorded in every resulting grid's other_candidates
    # so the raw evidence (offset/dtype/stats) stays inspectable regardless
    # of which ones got rasterised.
    other_candidates = []
    for c in cands:
        s = c.run.stats()
        if not s:
            continue
        other_candidates.append({
            "offset": int(c.run.offset), "dtype": c.run.dtype, "n": int(s["n"]),
            "score": round(float(c.score), 2), "min": round(float(s["min"]), 2),
            "max": round(float(s["max"]), 2), "mean": round(float(s["mean"]), 2),
            "zero_frac": round(s.get("zeros", 0) / s["n"], 3) if s["n"] else 0.0,
        })

    keep = [c for c in cands if c.score >= MULTI_GRID_SCORE_FLOOR][:MAX_GRIDS_PER_GROUP]
    if not keep:
        keep = cands[:1]  # always try at least the best one, even if weak

    grids: List[CalcGrid] = []
    for rank, cand in enumerate(keep):
        g = _build_grid(cand, rank, len(keep), group_name, name, illum.path,
                        clptr, room_aspect, room_extent, other_candidates)
        if g is not None and g.values is not None:
            grids.append(g)

    grids.sort(key=lambda g: -g.confidence)
    return grids


def _trim_padding(a: np.ndarray, max_frac: float = 0.02) -> np.ndarray:
    """Strip a short run of identical values (usually 0.0 padding) from either end."""
    n = a.size
    if n < 8:
        return a
    limit = max(2, int(n * max_frac))

    # Leading run of identical values
    lo = 0
    k = 1
    while k < n and a[k] == a[0]:
        k += 1
    if 2 <= k <= limit:
        lo = k

    # Trailing run of identical values
    hi = n
    k = 1
    while k < n and a[n - 1 - k] == a[-1]:
        k += 1
    if 2 <= k <= limit:
        hi = n - k

    return a[lo:hi] if (hi - lo) >= n // 2 else a


def _raster_quality(grid: np.ndarray) -> float:
    """
    Score a candidate raster. A correct one is fully populated and smooth;
    a misaligned one has holes and jagged neighbour differences.
    """
    if grid is None or grid.size < 4:
        return -1.0
    fill = float(np.isfinite(grid).mean())
    g = grid[np.isfinite(grid)]
    if g.size < 4:
        return -1.0
    spread = float(g.max() - g.min())
    if spread <= 0:
        return fill * 0.5
    d = np.abs(np.diff(np.nan_to_num(grid, nan=float(np.nanmean(grid))), axis=1))
    smooth = 1.0 - min(float(np.median(d)) / spread / 0.15, 1.0)
    square = min(grid.shape) / max(grid.shape)

    # Isolated extremes: a cell wildly out of step with its neighbours is the
    # signature of padding bytes that slipped into the array.
    filled = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
    pad = np.pad(filled, 1, mode="edge")
    neigh = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4.0
    outliers = float((np.abs(filled - neigh) > spread * 0.40).mean())

    # Exact 0.0 on a lit working plane is nearly always padding, not a reading.
    zero_frac = float((g == 0).mean())
    zero_pen = 1.0 - min(zero_frac * 50.0, 1.0) if g.mean() > 1.0 else 1.0

    return (fill * 0.40 + smooth * 0.25 + square * 0.10
            + (1.0 - min(outliers * 20.0, 1.0)) * 0.10
            + zero_pen * 0.15)


def _best_alignment(pts: np.ndarray, values: np.ndarray, room_extent=None):
    """
    Try head-aligned, tail-aligned and padding-trimmed pairings of the point
    and value arrays. Return (rasterise_result, points_used) for the best.

    Padding-trimmed variants are evaluated FIRST and win outright if they
    produce any valid raster. Trimming a run of identical values off the end
    of an array is a principled correction; the untrimmed array is only a
    fallback for when trimming leaves too few points to raster.
    """
    v_variants = [_trim_padding(values), values]
    pt_trimmed = pts
    if pts.size > 3:
        k = len(_trim_padding(pts.ravel())) // 3
        if 4 <= k <= len(pts):
            pt_trimmed = pts[:k]
    p_variants = [pt_trimmed, pts]

    for v in v_variants:
        best, best_pts, best_score = None, None, -1.0
        for p in p_variants:
            if p is None or len(p) < 4 or v.size < 4:
                continue
            n = min(len(p), v.size)
            for pa, va in ((p[:n], v[:n]), (p[-n:], v[-n:]),
                           (p[:n], v[-n:]), (p[-n:], v[:n])):
                try:
                    res = _rasterise(pa, va, room_extent)
                except Exception:
                    continue
                if res is None:
                    continue
                score = _raster_quality(res[0])
                if score > best_score:
                    best, best_pts, best_score = res, pa, score
        if best is not None and best_score > 0:
            return best, best_pts
    return None, None


def _extent_match(extent, room_extent) -> float:
    """How well a recovered raster extent matches the known room bounds."""
    if room_extent is None:
        return 0.0
    rx0, ry0, rx1, ry1 = room_extent
    ex0, ey0, ex1, ey1 = extent
    rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
    ew, eh = abs(ex1 - ex0), abs(ey1 - ey0)
    if rw <= 0 or rh <= 0 or ew <= 0 or eh <= 0:
        return 0.0
    # Compare both width/height ratio and absolute size
    ratio = min(ew / rw, rw / ew) * min(eh / rh, rh / eh)
    return float(ratio)


def _rasterise(pts: np.ndarray, values: np.ndarray, room_extent=None):
    """
    Turn scattered points + values into a regular 2-D raster.

    The point array has three columns but we do not know which one is the
    constant "height" axis — a misaligned read can make any column look
    constant. So every orientation is tried and scored, using the known room
    bounds as the tie-breaker when available.
    """
    n = min(len(pts), len(values))
    if n < 4:
        return None
    pts, values = pts[:n], values[:n]

    best, best_score = None, -1.0
    orientations = []
    for plane_axis in range(3):
        axes = [i for i in range(3) if i != plane_axis]
        # A misaligned read also permutes the columns, so which of the two
        # remaining axes is "x" is not known either. Try both.
        orientations.append((plane_axis, axes[0], axes[1]))
        orientations.append((plane_axis, axes[1], axes[0]))

    for plane_axis, ax_x, ax_y in orientations:
        X, Y = pts[:, ax_x], pts[:, ax_y]

        xs = np.unique(np.round(X, 4))
        ys = np.unique(np.round(Y, 4))
        if xs.size < 2 or ys.size < 2 or xs.size * ys.size > n * 4:
            continue

        grid = np.full((ys.size, xs.size), np.nan)
        count = np.zeros((ys.size, xs.size), dtype=int)
        xi = np.clip(np.searchsorted(xs, np.round(X, 4)), 0, xs.size - 1)
        yi = np.clip(np.searchsorted(ys, np.round(Y, 4)), 0, ys.size - 1)
        grid[yi, xi] = values
        count[yi, xi] += 1

        # Stray floats from a file header or trailer can invent a phantom row
        # or column holding one or two points. A real calculation raster is
        # fully populated, so drop anything less than half filled.
        row_ok = (count > 0).mean(axis=1) >= 0.5
        col_ok = (count > 0).mean(axis=0) >= 0.5
        if row_ok.any() and col_ok.any() and (not row_ok.all() or not col_ok.all()):
            grid = grid[np.ix_(row_ok, col_ok)]
            xs_k, ys_k = xs[col_ok], ys[row_ok]
        else:
            xs_k, ys_k = xs, ys

        if grid.size < 4:
            continue

        extent = (float(xs_k.min()), float(ys_k.min()),
                  float(xs_k.max()), float(ys_k.max()))

        # A genuine working plane is flat in the axis we excluded.
        flat = 1.0 - min(float(np.ptp(pts[:, plane_axis])) /
                         (float(np.ptp(pts).max()) or 1.0), 1.0) \
            if np.ptp(pts) > 0 else 0.0
        score = (_raster_quality(grid) * 0.55
                 + flat * 0.20
                 + _extent_match(extent, room_extent) * 0.25)

        if score > best_score:
            best, best_score = (grid, extent), score

    return best


def read_all_groups(groups: Dict[str, List[Entry]],
                    room_aspect: Optional[float] = None,
                    room_extent=None,
                    progress=None,
                    ) -> Tuple[List[CalcGrid], List["EnvironmentReading"]]:
    """
    Read every result group, dropping duplicates.

    evo routinely stores the same calculation twice — once under the room's
    own group and once under a parent/whole-project group — so identical
    result sets are collapsed into one. Returns (grids, environment_readings).

    `progress`, if given, is called as `progress(done, total, group_name)`
    after each group is processed — real counts, not a simulated ramp, so a
    caller (e.g. a GUI) can show an honest percentage for what is usually
    the slowest part of reading a project with many result groups.
    """
    from .model import EnvironmentReading

    out: List[CalcGrid] = []
    env: List[EnvironmentReading] = []
    seen_signatures = set()
    seen_env = set()

    items = sorted(groups.items())
    total = len(items)
    for i, (gname, entries) in enumerate(items):
        try:
            group_grids = read_group(gname, entries, room_aspect, room_extent)
        except Exception:
            group_grids = []
        for g in group_grids:
            if g is None or g.values is None:
                continue
            m = g.metrics()
            sig = (g.values.size, round(m.get("Eav", 0), 4),
                   round(m.get("Emax", 0), 4))
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                out.append(g)

        env_entry = next((e for e in entries
                          if "envillum" in e.name.lower()
                          and not e.name.lower().endswith("toc.rsl")), None)
        if env_entry is not None:
            vals = extract_scalars(env_entry.read())
            if vals:
                sig = tuple(vals)
                if sig not in seen_env:
                    seen_env.add(sig)
                    env.append(EnvironmentReading(
                        group=gname, values=vals, source=env_entry.path))

        if progress is not None:
            progress(i + 1, total, gname)

    out.sort(key=lambda g: -g.confidence)
    return out, env
