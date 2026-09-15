"""
Heuristic binary analysis for DIALux evo's undocumented payloads
(.rsl result files, ProjectData.dat, ScenegraphScene).

There is no published specification for these. What this module does instead
is treat them as haystacks and look for needles with known statistical
shape: long contiguous runs of IEEE floats whose values fall inside a
physically plausible range.

That is enough to recover the two things that actually matter:
  * illuminance value arrays  (Dataillumqt*.rsl)
  * calculation point coordinates (Dataclptr*.rsl)

Everything reported by this module is a *candidate* with a confidence score.
Nothing here is guaranteed correct, and the CLI always shows you the evidence
so you can sanity-check it against DIALux itself.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Plausibility windows ------------------------------------------------------
# Illuminance in lux: interior scenes run ~0.1 to a few thousand; outdoor
# floodlighting can reach tens of thousands. Anything beyond 500 klx is noise.
LUX_RANGE = (0.0, 500_000.0)
# Scene coordinates in metres. DIALux works in mm internally in places, so we
# also test a millimetre window.
METRE_RANGE = (-500.0, 500.0)
MM_RANGE = (-500_000.0, 500_000.0)


@dataclass
class FloatRun:
    """A contiguous stretch of bytes that decodes as plausible floats."""

    offset: int
    count: int
    dtype: str          # 'f4' / 'f8', with '<' or '>' byte order
    values: np.ndarray = field(repr=False)

    @property
    def nbytes(self) -> int:
        return self.count * np.dtype(self.dtype).itemsize

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def stats(self) -> dict:
        v = self.values
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return {}
        return {
            "n": int(finite.size),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "std": float(finite.std()),
            "zeros": int((finite == 0).sum()),
        }

    def describe(self) -> str:
        s = self.stats()
        if not s:
            return f"@{self.offset} x{self.count} {self.dtype} (all non-finite)"
        return (
            f"@0x{self.offset:06x}  n={s['n']:<7} {self.dtype}  "
            f"min={s['min']:<12.4g} max={s['max']:<12.4g} "
            f"mean={s['mean']:<12.4g} median={s['median']:.4g}"
        )


def _plausible_mask(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True where a value is finite, in range, and not a denormal."""
    with np.errstate(invalid="ignore"):
        m = np.isfinite(arr) & (arr >= lo) & (arr <= hi)
        tiny = (np.abs(arr) > 0) & (np.abs(arr) < 1e-8)
        return m & ~tiny


def find_float_runs(
    data: bytes,
    value_range: Tuple[float, float] = LUX_RANGE,
    dtypes: Tuple[str, ...] = ("<f4", "<f8", ">f4", ">f8"),
    min_count: int = 32,
    max_runs: int = 12,
) -> List[FloatRun]:
    """
    Scan `data` for contiguous runs of floats that all land inside
    `value_range`. Every byte alignment is tested for each dtype.

    Returns the longest runs first, de-overlapped.
    """
    lo, hi = value_range
    runs: List[FloatRun] = []

    for dt in dtypes:
        item = np.dtype(dt).itemsize
        for align in range(item):
            usable = (len(data) - align) // item * item
            if usable < min_count * item:
                continue
            arr = np.frombuffer(data, dtype=dt, count=usable // item, offset=align)
            mask = _plausible_mask(arr, lo, hi)
            if not mask.any():
                continue

            # Walk mask for maximal True runs
            idx = np.flatnonzero(
                np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
            )
            starts, ends = idx[0::2], idx[1::2]
            for s, e in zip(starts, ends):
                n = e - s
                if n < min_count:
                    continue
                runs.append(
                    FloatRun(
                        offset=int(align + s * item),
                        count=int(n),
                        dtype=dt,
                        values=arr[s:e].astype(np.float64),
                    )
                )

    runs.sort(key=lambda r: r.count, reverse=True)

    # Drop runs that sit inside a longer, already-accepted run
    kept: List[FloatRun] = []
    for r in runs:
        if any(r.offset >= k.offset and r.end <= k.end for k in kept):
            continue
        kept.append(r)
        if len(kept) >= max_runs:
            break
    return kept


def score_as_illuminance(run: FloatRun) -> float:
    """
    Confidence 0..1 that a run is an illuminance grid rather than incidental
    floats. Illuminance grids are: long, non-negative, smoothly varying,
    with a wide but not absurd dynamic range and few exact duplicates.
    """
    s = run.stats()
    if not s or s["n"] < 64:
        return 0.0
    score = 0.0
    v = run.values

    # Long arrays are far more likely to be the payload
    score += min(s["n"] / 4000.0, 1.0) * 0.30
    # Non-negative
    if s["min"] >= 0:
        score += 0.15
    # Sensible interior/exterior magnitudes
    if 1.0 <= s["max"] <= 200_000.0:
        score += 0.20
    if 0.5 <= s["mean"] <= 50_000.0:
        score += 0.10
    # Smoothness: neighbouring grid points correlate strongly
    if v.size > 8:
        d = np.abs(np.diff(v))
        spread = s["max"] - s["min"]
        if spread > 0 and np.median(d) < spread * 0.10:
            score += 0.15
    # Not a run of identical padding
    uniq = np.unique(v).size / v.size
    if uniq > 0.25:
        score += 0.10
    return min(score, 1.0)


def score_as_coordinates(run: FloatRun) -> float:
    """Confidence that a run is XYZ point coordinates."""
    s = run.stats()
    if not s or s["n"] < 30:
        return 0.0
    score = 0.0
    # Coordinate arrays usually come in multiples of 2 or 3
    if s["n"] % 3 == 0:
        score += 0.25
    elif s["n"] % 2 == 0:
        score += 0.15
    # Building-scale magnitudes
    if abs(s["max"]) < 1000 and abs(s["min"]) < 1000:
        score += 0.25
    # Regular grids show a small set of distinct values per axis
    v = run.values
    uniq = np.unique(np.round(v, 4)).size
    if 0 < uniq < v.size * 0.6:
        score += 0.25
    # Monotone-ish sweeps are typical of a raster grid
    if v.size > 3 and (np.diff(v) >= 0).mean() > 0.7:
        score += 0.25
    return min(score, 1.0)


def detect_stride(data: bytes, max_stride: int = 512) -> Optional[int]:
    """
    Guess a fixed record size by autocorrelating the byte stream.
    Record-structured files (the *Toc.rsl indexes) show a strong peak.
    """
    if len(data) < max_stride * 4:
        return None
    a = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    a = a - a.mean()
    best, best_score = None, 0.0
    for stride in range(4, min(max_stride, len(a) // 4)):
        x, y = a[:-stride], a[stride:]
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom == 0:
            continue
        score = float(np.dot(x, y) / denom)
        if score > best_score:
            best, best_score = stride, score
    return best if best_score > 0.55 else None


def find_ascii_strings(data: bytes, min_len: int = 4, limit: int = 200
                       ) -> List[Tuple[int, str]]:
    """Pull readable ASCII and UTF-16LE strings out of a binary blob."""
    out: List[Tuple[int, str]] = []

    # ASCII
    cur, start = [], 0
    for i, b in enumerate(data):
        if 32 <= b <= 126:
            if not cur:
                start = i
            cur.append(chr(b))
        else:
            if len(cur) >= min_len:
                out.append((start, "".join(cur)))
            cur = []
    if len(cur) >= min_len:
        out.append((start, "".join(cur)))

    # UTF-16LE (very common in .NET serialisation)
    i = 0
    while i < len(data) - 1:
        if 32 <= data[i] <= 126 and data[i + 1] == 0:
            j, chars = i, []
            while j < len(data) - 1 and 32 <= data[j] <= 126 and data[j + 1] == 0:
                chars.append(chr(data[j]))
                j += 2
            if len(chars) >= min_len:
                out.append((i, "".join(chars) + "  [utf-16]"))
            i = j
        else:
            i += 1

    out.sort(key=lambda t: t[0])
    return out[:limit]


def hexdump(data: bytes, offset: int = 0, length: int = 256) -> str:
    """Classic hex + ASCII dump, for eyeballing headers."""
    lines = []
    chunk = data[offset:offset + length]
    for i in range(0, len(chunk), 16):
        row = chunk[i:i + 16]
        hexpart = " ".join(f"{b:02x}" for b in row).ljust(47)
        txt = "".join(chr(b) if 32 <= b <= 126 else "." for b in row)
        lines.append(f"{offset + i:08x}  {hexpart}  |{txt}|")
    return "\n".join(lines)


def infer_grid_shape(n: int, aspect: Optional[float] = None
                     ) -> Optional[Tuple[int, int]]:
    """
    Given n calculation values, find the (nx, ny) factor pair whose aspect
    ratio best matches the room. Without a room aspect, prefer the most
    square factorisation.
    """
    if n < 4:
        return None
    pairs = [(i, n // i) for i in range(1, int(n ** 0.5) + 1) if n % i == 0]
    if not pairs:
        return None
    if aspect and aspect > 0:
        best = min(pairs, key=lambda p: abs((p[1] / p[0]) - aspect))
        # Try both orientations
        alt = min(pairs, key=lambda p: abs((p[0] / p[1]) - aspect))
        if abs((alt[0] / alt[1]) - aspect) < abs((best[1] / best[0]) - aspect):
            return (alt[1], alt[0])
        return best
    return max(pairs, key=lambda p: p[0])
