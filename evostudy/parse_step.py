"""
Reader for DIALux evo's Project/ProjectData/ProjectData.dat.

Unlike the .rsl result files, this one is NOT an undocumented binary blob.
It decodes as plain UTF-8 text and turns out to be an ISO-10303-21 file
("STEP", Clear Text Encoding of the Exchange Structure) — a real, public
ISO standard container format, most familiar from CAD/BIM interchange
(the IFC format is built the same way). Its DATA section is a flat graph
of typed entity records:

    #43 = LuminaireElement (115, 'guid...', (#399, #400, #401), #402, #403, #404);

`#43` is this record's id; other records reference it as `#43`. DIALux's
own EXPRESS schema (the type definitions for each record) isn't published,
so which positional argument means what is still something we work out
from evidence rather than a spec — but the CONTAINER (where one record
ends, how references work, how strings/enums/nulls are encoded) is fully
deterministic. That is a much stronger starting point than the .rsl files:
no statistical guessing is needed to find record boundaries or follow a
reference, only to interpret an unfamiliar type's fields.

On ProjectData.xml-only evo builds (see parse_xml.py's warning), this file
is where the actual luminaire/room instances live — ProjectData.xml is
just the .NET type registry that declares the classes this file's records
are instances of.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .model import Luminaire

REF_RE = re.compile(r"#(\d+)")
RECORD_START_RE = re.compile(r"#(\d+)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
STRING_RE = re.compile(r"'((?:[^']|'')*)'")


@dataclass
class StepEntity:
    id: int
    type: str
    args: str          # raw, unparsed argument text between the outer parens

    def refs(self) -> List[int]:
        """Every #id referenced anywhere in this entity's arguments."""
        return [int(m.group(1)) for m in REF_RE.finditer(self.args)]

    def top_args(self) -> List[str]:
        """Split top-level comma-separated arguments, respecting nested
        parens and quoted strings (STEP strings escape a quote as '')."""
        out, buf, depth, in_str = [], [], 0, False
        i = 0
        s = self.args
        while i < len(s):
            c = s[i]
            if in_str:
                if c == "'" and s[i:i + 2] != "''":
                    in_str = False
                elif c == "'" and s[i:i + 2] == "''":
                    buf.append("''")
                    i += 2
                    continue
                buf.append(c)
            else:
                if c == "'":
                    in_str = True
                    buf.append(c)
                elif c == "(":
                    depth += 1
                    buf.append(c)
                elif c == ")":
                    depth -= 1
                    buf.append(c)
                elif c == "," and depth == 0:
                    out.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(c)
            i += 1
        if buf:
            out.append("".join(buf).strip())
        return out


def parse_step_entities(text: str) -> Dict[int, StepEntity]:
    """
    Parse every `#id = TypeName(...)` record in a STEP file's DATA section
    into a flat {id: StepEntity} graph. Depth-counts parens and respects
    quoted strings, so it's not fooled by nested tuples or a stray ')'
    inside a text field — deterministic, not a heuristic.
    """
    entities: Dict[int, StepEntity] = {}
    n = len(text)
    for m in RECORD_START_RE.finditer(text):
        eid = int(m.group(1))
        etype = m.group(2)
        start = m.end()
        depth = 1
        in_str = False
        j = start
        while j < n and depth > 0:
            c = text[j]
            if in_str:
                if c == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 1
                    else:
                        in_str = False
            else:
                if c == "'":
                    in_str = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            j += 1
        entities[eid] = StepEntity(id=eid, type=etype, args=text[start:j - 1])
    return entities


def _parse_triplet(text: str) -> Optional[Tuple[float, float, float]]:
    """'(1.0, 2.0, 3.0)' -> (1.0, 2.0, 3.0)"""
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text)
    if len(nums) < 3:
        return None
    try:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    except ValueError:
        return None


def _find_coordsys(entities: Dict[int, StepEntity], eid: int
                   ) -> Optional[Tuple[Tuple[float, float, float],
                                        Tuple[float, float, float],
                                        Tuple[float, float, float]]]:
    """Return (origin, x_axis, y_axis) for a CoordSys3D-typed entity."""
    e = entities.get(eid)
    if e is None or e.type != "CoordSys3D":
        return None
    parts = e.top_args()
    if len(parts) < 3:
        return None
    origin = _parse_triplet(parts[0])
    x_axis = _parse_triplet(parts[1])
    y_axis = _parse_triplet(parts[2])
    if origin is None or x_axis is None or y_axis is None:
        return None
    return origin, x_axis, y_axis


def _bfs_find_type(entities: Dict[int, StepEntity], start_ids: List[int],
                   want_type: str, max_depth: int = 5, max_visit: int = 400
                   ) -> Optional[StepEntity]:
    """Breadth-first search the reference graph from `start_ids` for the
    first entity of type `want_type`, without assuming a fixed field
    layout — robust to schema differences between evo versions."""
    seen = set(start_ids)
    frontier = list(start_ids)
    depth = 0
    visited_count = 0
    while frontier and depth < max_depth and visited_count < max_visit:
        nxt = []
        for eid in frontier:
            e = entities.get(eid)
            visited_count += 1
            if e is None:
                continue
            if e.type == want_type:
                return e
            for r in e.refs():
                if r not in seen:
                    seen.add(r)
                    nxt.append(r)
        frontier = nxt
        depth += 1
    return None


def _extract_article_name(entities: Dict[int, StepEntity],
                          text_container_ref: Optional[int]) -> str:
    if text_container_ref is None:
        return ""
    e = entities.get(text_container_ref)
    if e is None or e.type != "LanguageDependentTextContainer":
        return ""
    m = re.search(r"\.ArticleName\.\s*,\s*'((?:[^']|'')*)'", e.args)
    if not m:
        return ""
    return m.group(1).replace("''", "'").replace("%2E", ".").replace("%2C", ",")


def _resolve_product(entities: Dict[int, StepEntity], prototype_id: int,
                     cache: Dict[int, Tuple[str, str]]) -> Tuple[str, str]:
    """(manufacturer, article_name) for a LuminairePrototype id, memoised
    since many placed luminaires typically share one prototype."""
    if prototype_id in cache:
        return cache[prototype_id]
    proto = entities.get(prototype_id)
    manufacturer, article = "", ""
    if proto is not None:
        pdata = _bfs_find_type(entities, proto.refs(), "ProductData")
        if pdata is not None:
            args = pdata.top_args()
            if args:
                sm = STRING_RE.match(args[0])
                manufacturer = sm.group(1).replace("''", "'") if sm else ""
            text_ref = None
            for a in args:
                m = re.match(r"^#(\d+)$", a.strip())
                if m:
                    text_ref = int(m.group(1))
                    break
            article = _extract_article_name(entities, text_ref)
    cache[prototype_id] = (manufacturer, article)
    return manufacturer, article


def extract_luminaires(text: str) -> Tuple[List[Luminaire], List[str]]:
    """
    Recover placed luminaire instances (position + product identity) from
    a decoded ProjectData.dat STEP file.

    Returns (luminaires, warnings). Position (from each instance's own
    CoordSys3D) is read directly off the entity graph — deterministic.
    Manufacturer/article name requires walking an unpublished schema by
    reference-following (`_bfs_find_type`), so treat those two fields as
    recovered-with-moderate-confidence: cross-check a few against DIALux's
    own luminaire schedule before trusting article numbers for ordering.
    """
    warnings: List[str] = []
    entities = parse_step_entities(text)
    if not entities:
        return [], ["ProjectData.dat did not parse as a STEP entity graph "
                    "(no '#id = Type(...)' records found)."]

    lum_ids = [eid for eid, e in entities.items() if e.type == "LuminaireElement"]
    if not lum_ids:
        return [], []

    # Map LuminaireElement id -> LuminairePrototype id via RelDefinesByPrototype
    # records: (..., ((#lum_a, #x), (#lum_b, #y), ...), #prototype_id).
    proto_of: Dict[int, int] = {}
    for e in entities.values():
        if e.type != "RelDefinesByPrototype":
            continue
        args = e.top_args()
        if len(args) < 2:
            continue
        proto_m = re.search(r"#(\d+)\s*$", args[-1])
        if not proto_m:
            continue
        proto_id = int(proto_m.group(1))
        pairs = re.findall(r"\(#(\d+)\s*,\s*#\d+\)", args[-2] if len(args) >= 2 else "")
        for lum_id_str in pairs:
            proto_of[int(lum_id_str)] = proto_id

    product_cache: Dict[int, Tuple[str, str]] = {}
    luminaires: List[Luminaire] = []
    n_no_position, n_no_product = 0, 0

    for eid in sorted(lum_ids):
        e = entities[eid]
        args = e.top_args()
        guid = ""
        if len(args) > 1:
            gm = STRING_RE.match(args[1])
            guid = gm.group(1) if gm else ""

        # Position: search this entity's directly-referenced ids for a
        # CoordSys3D (its own placement is normally one hop away).
        cs = None
        for r in e.refs():
            cs = _find_coordsys(entities, r)
            if cs is not None:
                break
        x = y = z = 0.0
        rotation = 0.0
        if cs is not None:
            (ox, oy, oz), (xx, xy, xz), (yx, yy, yz) = cs
            x, y, z = ox, oy, oz
            rotation = math.degrees(math.atan2(xy, xx)) % 360.0
        else:
            n_no_position += 1

        manufacturer, article = "", ""
        proto_id = proto_of.get(eid)
        if proto_id is not None:
            manufacturer, article = _resolve_product(entities, proto_id, product_cache)
        if not manufacturer and not article:
            n_no_product += 1

        luminaires.append(Luminaire(
            id=guid or str(eid), name=article, manufacturer=manufacturer,
            x=x, y=y, z=z, rotation=rotation, count=1,
            raw={"step_id": eid, "position_recovered": cs is not None,
                "product_recovered": bool(manufacturer or article)},
        ))

    if n_no_position:
        warnings.append(
            f"{n_no_position} of {len(luminaires)} recovered luminaire(s) had "
            f"no CoordSys3D reachable one hop away — their x/y/z default to "
            f"0.0 and should not be trusted.")
    if n_no_product:
        warnings.append(
            f"{n_no_product} of {len(luminaires)} recovered luminaire(s) "
            f"could not be traced to a ProductData record (manufacturer/"
            f"article left blank) — the prototype-to-product reference "
            f"chain may differ on this evo version.")
    warnings.append(
        "Luminaire positions/rotation come directly from this file's "
        "CoordSys3D records (deterministic). Manufacturer/article name are "
        "recovered by following an unpublished reference chain "
        "(LuminaireElement -> RelDefinesByPrototype -> LuminairePrototype "
        "-> ProductData) — cross-check a few entries against DIALux's own "
        "luminaire schedule before trusting article numbers for ordering.")
    return luminaires, warnings
