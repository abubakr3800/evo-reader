"""
Adaptive parser for ProjectData.xml / ArchiveInfo.xml / OutputSettings.xml.

DIALux does not publish its schema, and tag names differ between evo
versions and between German and English builds. Rather than hard-coding
one guess, this parser is driven by a keyword MAPPING that you can override
with a JSON file (`--mapping my_tags.json`), and it falls back to structural
heuristics when keywords miss.

Run `evostudy schema <archive>` first: it prints the real tag tree of your
file so you can see exactly what to map.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .model import CalcGrid, Luminaire, Polygon, Room, Study

# ---------------------------------------------------------------------------
# Keyword mapping. Matching is case-insensitive substring against the tag's
# local name and against attribute names. German terms included because evo
# writes German internals in some builds.
# ---------------------------------------------------------------------------

DEFAULT_MAPPING: Dict[str, List[str]] = {
    "luminaire":   ["luminaire", "leuchte", "lamp", "light", "fixture"],
    "room":        ["room", "raum", "space", "zone"],
    "storey":      ["storey", "story", "floor", "geschoss", "level"],
    "building":    ["building", "gebaeude", "gebäude", "site"],
    "polygon":     ["polygon", "outline", "contour", "profile", "boundary",
                    "umriss", "shape", "footprint"],
    "point":       ["point", "vertex", "vertice", "corner", "punkt", "node",
                    "coordinate"],
    "calcsurface": ["calculationsurface", "calcsurface", "workingplane",
                    "workplane", "nutzebene", "utilisation", "surface",
                    "calculationobject", "berechnungsflaeche"],
    "name":        ["name", "designation", "title", "label", "bezeichnung",
                    "description"],
    "manufacturer": ["manufacturer", "hersteller", "brand", "vendor"],
    "article":     ["article", "articleno", "ordernumber", "artikel",
                    "partnumber", "productnumber", "typ"],
    "flux":        ["luminousflux", "lumen", "flux", "lichtstrom", "phi"],
    "power":       ["power", "wattage", "watt", "leistung", "connectedload",
                    "p_total"],
    "cct":         ["colourtemperature", "colortemperature", "cct",
                    "farbtemperatur", "kelvin"],
    "cri":         ["cri", "ra", "colourrendering", "colorrendering",
                    "farbwiedergabe"],
    "maintenance": ["maintenancefactor", "wartungsfaktor", "mf", "lldf"],
    "reflectance": ["reflect", "reflexionsgrad", "rho"],
    "height":      ["height", "hoehe", "höhe", "z"],
    "project":     ["project", "projekt"],
    "author":      ["author", "operator", "planner", "verfasser", "bearbeiter"],
    "company":     ["company", "firma", "office"],
}

NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


def load_mapping(path: Optional[str]) -> Dict[str, List[str]]:
    mapping = {k: list(v) for k, v in DEFAULT_MAPPING.items()}
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in user.items():
            mapping[k] = list(v) if isinstance(v, list) else [str(v)]
    return mapping


# -- small helpers ----------------------------------------------------------

def localname(tag: str) -> str:
    """Strip any XML namespace."""
    return tag.split("}")[-1] if "}" in tag else tag


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matches(token: str, keywords: Iterable[str]) -> bool:
    t = norm(token)
    return any(norm(k) in t for k in keywords)


def to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if NUM_RE.match(s):
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return None
    # Values like "3200 lm" or "36 W"
    m = re.match(r"^(-?\d+(?:[.,]\d+)?)", s)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def gather_fields(el: ET.Element, depth: int = 2) -> Dict[str, str]:
    """
    Flatten an element's attributes plus its shallow descendants' attributes
    and text into one {key: value} dict, so keyword matching can work without
    knowing the nesting.
    """
    out: Dict[str, str] = {}

    def walk(e: ET.Element, d: int):
        for k, v in e.attrib.items():
            out.setdefault(localname(k), v)
        if e.text and e.text.strip():
            out.setdefault(localname(e.tag), e.text.strip())
        if d <= 0:
            return
        for c in e:
            walk(c, d - 1)

    walk(el, depth)
    return out


def pick(fields: Dict[str, str], keywords: Iterable[str]) -> Optional[str]:
    """First field whose key matches one of the keywords."""
    for k, v in fields.items():
        if matches(k, keywords):
            return v
    return None


def pick_float(fields: Dict[str, str], keywords: Iterable[str]) -> Optional[float]:
    return to_float(pick(fields, keywords))


def xyz(fields: Dict[str, str]) -> Tuple[float, float, float]:
    """Pull an x/y/z triple out of a flattened field dict."""
    def one(*names):
        for n in names:
            for k, v in fields.items():
                if norm(k) == n:
                    f = to_float(v)
                    if f is not None:
                        return f
        return None

    x = one("x", "posx", "positionx", "cx")
    y = one("y", "posy", "positiony", "cy")
    z = one("z", "posz", "positionz", "cz", "height", "hoehe")

    # Some builds write "1.20 3.40 2.80" in a single attribute
    if x is None:
        for k, v in fields.items():
            if matches(k, ["position", "location", "translation", "origin"]):
                nums = [to_float(t) for t in re.split(r"[;,\s]+", v)]
                nums = [n for n in nums if n is not None]
                if len(nums) >= 2:
                    x, y = nums[0], nums[1]
                    z = nums[2] if len(nums) > 2 else z
                    break
    return (x or 0.0, y or 0.0, z or 0.0)


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def schema_report(xml_bytes: bytes, max_lines: int = 400) -> str:
    """
    Print the element tree as path -> count, with the attribute names seen at
    each path. This is the file you read before tuning a mapping.
    """
    root = ET.fromstring(xml_bytes)
    counts: Counter = Counter()
    attrs: Dict[str, set] = defaultdict(set)
    samples: Dict[str, str] = {}

    def walk(el: ET.Element, path: str):
        p = f"{path}/{localname(el.tag)}"
        counts[p] += 1
        for k, v in el.attrib.items():
            attrs[p].add(localname(k))
            samples.setdefault(f"{p}@{localname(k)}", v)
        for c in el:
            walk(c, p)

    walk(root, "")

    lines = [f"root element: {localname(root.tag)}",
             f"distinct paths: {len(counts)}", ""]
    for path, n in sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:max_lines]:
        a = sorted(attrs.get(path, []))
        a_txt = ""
        if a:
            shown = a[:8]
            ex = []
            for k in shown:
                s = samples.get(f"{path}@{k}", "")
                if len(s) > 18:
                    s = s[:18] + "…"
                ex.append(f"{k}={s!r}")
            a_txt = "   attrs: " + ", ".join(ex)
            if len(a) > 8:
                a_txt += f" (+{len(a) - 8} more)"
        lines.append(f"{n:>6}  {path}{a_txt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

class ProjectParser:
    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None):
        self.m = mapping or DEFAULT_MAPPING
        self.warnings: List[str] = []

    # -- entry point ---------------------------------------------------------

    def parse(self, project_xml: bytes,
              archive_info: Optional[bytes] = None) -> Study:
        study = Study()
        if archive_info:
            self._parse_archive_info(archive_info, study)

        try:
            root = ET.fromstring(project_xml)
        except ET.ParseError as e:
            self.warnings.append(f"ProjectData.xml did not parse: {e}")
            study.warnings = self.warnings
            return study

        elements = list(root.iter())
        self._parse_project_meta(root, study)
        self._parse_luminaires(elements, study)
        self._parse_rooms(elements, study)
        self._parse_geometry(elements, study)

        if not study.luminaires:
            self.warnings.append(
                "No luminaires matched. Run `evostudy schema` and add the real "
                "tag names to a mapping JSON under the 'luminaire' key.")
        if not study.rooms and not study.geometry:
            self.warnings.append(
                "No room outline matched. Add the real tag names under "
                "'room' / 'polygon' / 'point' in your mapping JSON.")

        study.warnings.extend(self.warnings)
        return study

    # -- pieces --------------------------------------------------------------

    def _parse_archive_info(self, data: bytes, study: Study) -> None:
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return
        f = gather_fields(root, depth=4)
        # "version" alone matches the archive schema version too eagerly;
        # prefer the more specific application-version field first.
        study.dialux_version = (
            pick(f, ["applicationversion", "dialuxversion", "appversion"])
            or pick(f, ["version"])
            or study.dialux_version)
        study.created = pick(f, ["created", "date", "timestamp", "datum"]) or ""
        study.project_name = pick(f, self.m["name"]) or study.project_name

    def _parse_project_meta(self, root: ET.Element, study: Study) -> None:
        # Look at the outermost few levels only — metadata lives near the top
        f: Dict[str, str] = {}
        for el in list(root.iter())[:60]:
            for k, v in el.attrib.items():
                f.setdefault(localname(k), v)
        study.project_name = (pick(f, self.m["name"]) or study.project_name
                              or "DIALux evo project")
        study.author = pick(f, self.m["author"]) or study.author
        study.company = pick(f, self.m["company"]) or study.company
        study.description = pick(f, ["description", "comment", "beschreibung"]) or ""

    def _parse_luminaires(self, elements: List[ET.Element], study: Study) -> None:
        seen = set()
        for el in elements:
            tag = localname(el.tag)
            if not matches(tag, self.m["luminaire"]):
                continue
            # Skip container/plural wrappers with no own data
            if not el.attrib and len(el) > 3:
                continue
            f = gather_fields(el, depth=3)
            x, y, z = xyz(f)
            lid = f.get("id") or f.get("Id") or f.get("guid") or ""
            key = (lid, round(x, 4), round(y, 4), round(z, 4), tag)
            if key in seen:
                continue
            seen.add(key)

            lum = Luminaire(
                id=str(lid),
                name=pick(f, self.m["name"]) or "",
                manufacturer=pick(f, self.m["manufacturer"]) or "",
                article_no=pick(f, self.m["article"]) or "",
                x=x, y=y, z=z,
                rotation=pick_float(f, ["rotation", "rotz", "angle", "drehung",
                                        "orientation"]) or 0.0,
                tilt=pick_float(f, ["tilt", "neigung", "rotx", "pitch"]) or 0.0,
                luminous_flux=pick_float(f, self.m["flux"]),
                power=pick_float(f, self.m["power"]),
                cct=pick_float(f, self.m["cct"]),
                cri=pick_float(f, self.m["cri"]),
                count=int(pick_float(f, ["count", "quantity", "anzahl",
                                         "number"]) or 1),
                raw=f,
            )
            if lum.luminous_flux and lum.power:
                lum.efficacy = lum.luminous_flux / lum.power
            study.luminaires.append(lum)

    def _parse_rooms(self, elements: List[ET.Element], study: Study) -> None:
        for el in elements:
            tag = localname(el.tag)
            if not matches(tag, self.m["room"]):
                continue
            f = gather_fields(el, depth=2)
            poly = self._first_polygon(el)
            room = Room(
                id=str(f.get("id", "")),
                name=pick(f, self.m["name"]) or f"Room {len(study.rooms)+1}",
                outline=poly,
                height=pick_float(f, self.m["height"]),
                maintenance_factor=pick_float(f, self.m["maintenance"]),
                raw=f,
            )
            refl = [to_float(v) for k, v in f.items()
                    if matches(k, self.m["reflectance"])]
            refl = [r for r in refl if r is not None]
            if len(refl) >= 3:
                room.reflectance_ceiling, room.reflectance_walls, \
                    room.reflectance_floor = refl[0], refl[1], refl[2]
            study.rooms.append(room)

    def _first_polygon(self, el: ET.Element) -> Optional[Polygon]:
        """Find the first plausible outline under an element."""
        for sub in el.iter():
            if matches(localname(sub.tag), self.m["polygon"]):
                p = self._points_of(sub)
                if p and len(p.points) >= 3:
                    return p
        p = self._points_of(el)
        return p if p and len(p.points) >= 3 else None

    def _points_of(self, el: ET.Element) -> Optional[Polygon]:
        pts: List[Tuple[float, float]] = []
        for sub in el.iter():
            if sub is el:
                continue
            if matches(localname(sub.tag), self.m["point"]):
                f = gather_fields(sub, depth=1)
                x, y, _ = xyz(f)
                pts.append((x, y))
        return Polygon(points=pts) if pts else None

    def _parse_geometry(self, elements: List[ET.Element], study: Study) -> None:
        """Collect any standalone outlines not already attached to a room."""
        claimed = {id(r.outline) for r in study.rooms if r.outline}
        for el in elements:
            if not matches(localname(el.tag), self.m["polygon"]):
                continue
            p = self._points_of(el)
            if p and len(p.points) >= 3 and id(p) not in claimed:
                p.name = localname(el.tag)
                study.geometry.append(p)
        # Deduplicate identical outlines
        uniq, seen = [], set()
        for p in study.geometry:
            key = tuple(round(c, 3) for pt in p.points for c in pt)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        study.geometry = uniq


def type_registry(xml_bytes: bytes) -> List[dict]:
    """
    Parse a MetaInfo/PersistenceObject type registry (the schema some evo
    versions ship instead of literal project data in ProjectData.xml).

    Returns one dict per PersistenceObject:
      {"name": str, "members": [{"index": str, "name": str, "type": str}]}
    """
    root = ET.fromstring(xml_bytes)
    objects = []
    for obj in root.iter():
        if localname(obj.tag) != "PersistenceObject":
            continue
        name = obj.get("Name", "")
        members = []
        for mem in obj:
            if localname(mem.tag) != "Member":
                continue
            idx = mem.get("IndexInObject", "")
            mname = mem.get("Name", "")
            # The outermost Type child names this member's .NET type. Deeper
            # nested <Type> elements are the serializer's shared type-table
            # cross references, not this member's own generic arguments, so
            # only the first level is meaningful for a field list.
            tname = ""
            for t in mem:
                if localname(t.tag) == "Type":
                    tname = t.get("Name", "")
                    break
            members.append({"index": idx, "name": mname, "type": tname})
        members.sort(key=lambda m: int(m["index"]) if str(m["index"]).isdigit()
                    else 0)
        objects.append({"name": name, "members": members})
    return objects


def type_registry_report(xml_bytes: bytes, filter_name: Optional[str] = None
                         ) -> str:
    """Human-readable dump of the type registry, for schema reconnaissance."""
    objects = type_registry(xml_bytes)
    if filter_name:
        fl = filter_name.lower()
        matches = [o for o in objects if fl in o["name"].lower()]
        if not matches:
            names = ", ".join(o["name"] for o in objects[:40])
            return (f"No PersistenceObject name contains {filter_name!r}.\n"
                    f"First 40 object names found:\n  {names}")
        lines = []
        for o in matches:
            lines.append(f"\n{o['name']}  ({len(o['members'])} members)")
            for m in o["members"]:
                lines.append(f"  [{m['index']:>3}] {m['name']:<28} {m['type']}")
        return "\n".join(lines)

    lines = [f"{len(objects)} PersistenceObject types found.\n",
             f"{'NAME':<50}{'MEMBERS':>8}"]
    lines.append("-" * 60)
    for o in sorted(objects, key=lambda o: o["name"]):
        lines.append(f"{o['name']:<50}{len(o['members']):>8}")
    lines.append("\nRe-run with --object <substring> to see one type's full "
                "field list, e.g.:")
    lines.append("  evostudy typeschema Project10.evo --object luminaire")
    return "\n".join(lines)
