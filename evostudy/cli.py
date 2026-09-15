"""
Command line interface.

    python -m evostudy inspect  Project10.evo
    python -m evostudy schema   Project10.evo
    python -m evostudy probe    Project10.evo
    python -m evostudy render   Project10.evo -o out/
    python -m evostudy export   Project10.evo -o out/ --format csv,json
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .archive import EvoArchive, sniff
from .binfmt import detect_stride, find_ascii_strings, hexdump
from .parse_rsl import analyse_rsl
from .parse_xml import schema_report, type_registry_report
from .pipeline import (
    export_csv, export_dashboard, export_json, export_pdf, export_per_fixture,
    export_pngs, load_study,
)

BANNER = "evostudy — DIALux evo project reader & study renderer"


# ---------------------------------------------------------------------------

def cmd_inspect(args) -> int:
    arc = EvoArchive.open(args.archive)
    print(BANNER)
    print(f"\nArchive : {args.archive}  ({arc.kind}, {len(arc)} files)\n")

    print(f"{'PATH':<58}{'SIZE':>10}  {'TYPE':<10} DESCRIPTION")
    print("-" * 110)
    for e in arc:
        ext, desc = e.sniff()
        print(f"{e.path:<58}{e.size:>10,}  {ext:<10} {desc}")

    groups = arc.result_groups()
    print(f"\nResult groups: {len(groups)}")
    for g, entries in groups.items():
        total = sum(e.size for e in entries)
        print(f"  {g:<40} {len(entries):>3} files, {total:>9,} bytes")

    if arc.archive_info:
        print("\n--- ArchiveInfo.xml ---")
        print(arc.archive_info.read().decode("utf-8", "replace")[:2000])
    return 0


def cmd_typeschema(args) -> int:
    """
    For evo versions where ProjectData.xml is a PersistenceObject type
    registry (a serializer schema) rather than literal project data: list
    the object types it defines, or show one type's field layout in full.
    """
    arc = EvoArchive.open(args.archive)
    e = arc.project_xml
    if e is None:
        print("Project/ProjectData/ProjectData.xml not found.", file=sys.stderr)
        return 2
    try:
        print(type_registry_report(e.read(), filter_name=args.object))
    except ET.ParseError as exc:
        print(f"could not parse as XML: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_schema(args) -> int:
    arc = EvoArchive.open(args.archive)
    targets = []
    if args.file:
        e = arc.get(args.file)
        if e is None:
            print(f"not found: {args.file}", file=sys.stderr)
            return 2
        targets = [e]
    else:
        targets = [t for t in (arc.project_xml, arc.output_settings,
                               arc.archive_info) if t]

    for e in targets:
        print(f"\n{'='*100}\n{e.path}   ({e.size:,} bytes)\n{'='*100}")
        try:
            print(schema_report(e.read(), max_lines=args.lines))
        except Exception as exc:
            print(f"  could not parse as XML: {exc}")
    print("\nCopy the real tag names into a mapping JSON, e.g.:")
    print('  {"luminaire": ["MyLuminaireTag"], "point": ["Vertex3D"]}')
    print("then re-run with:  --mapping mytags.json")
    return 0


def cmd_probe(args) -> int:
    """Look inside the binary payloads without committing to an interpretation."""
    arc = EvoArchive.open(args.archive)

    if args.file:
        entries = [arc.get(args.file)]
        if entries[0] is None:
            print(f"not found: {args.file}", file=sys.stderr)
            return 2
    else:
        entries = arc.find(suffix=".rsl")
        if arc.scenegraph:
            entries.append(arc.scenegraph)
        if arc.project_dat:
            entries.append(arc.project_dat)

    for e in entries:
        data = e.read()
        ext, desc = sniff(data)
        print(f"\n{'='*100}\n{e.path}   {e.size:,} bytes   [{ext}: {desc}]\n{'='*100}")

        print(hexdump(data, 0, 96))

        stride = detect_stride(data)
        if stride:
            print(f"\n  probable fixed record size: {stride} bytes "
                  f"({len(data)/stride:.1f} records)")

        strings = find_ascii_strings(data, min_len=5, limit=12)
        if strings:
            print("\n  embedded strings:")
            for off, s in strings:
                print(f"    @0x{off:06x}  {s[:80]}")

        kind = "coordinates" if "clptr" in e.name.lower() else "illuminance"
        cands = analyse_rsl(data, kind=kind)
        if cands:
            print(f"\n  float-array candidates (scored as {kind}):")
            for c in cands[:args.top]:
                print(f"    score {c.score:.2f}  {c.run.describe()}")
                v = c.run.values
                head = ", ".join(f"{x:.1f}" for x in v[:8])
                print(f"                first values: {head} ...")
        else:
            print("\n  no plausible float arrays found")
    return 0


def cmd_dashboard(args) -> int:
    study = load_study(args.archive, args.mapping, verbose=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    html = export_dashboard(study, out / "dashboard.html")
    print(f"\nDashboard : {html}")
    print("Open that file in any web browser — double-click it, no install "
         "or internet connection needed.")
    if args.per_fixture:
        files = export_per_fixture(study, out / "per_fixture_data")
        print(f"Per-fixture CSV+PNG : {len(files)} files in "
             f"{out / 'per_fixture_data'}")
    return 0


def cmd_render(args) -> int:
    study = load_study(args.archive, args.mapping, verbose=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdf = export_pdf(study, out / "study.pdf", include_3d=not args.no_3d)
    print(f"\nPDF report : {pdf}")

    html = export_dashboard(study, out / "dashboard.html")
    print(f"Dashboard  : {html}  (open in any browser)")

    if args.png:
        pngs = export_pngs(study, out / "sheets", dpi=args.dpi)
        print(f"PNG sheets : {len(pngs)} files in {out / 'sheets'}")

    if args.per_fixture:
        files = export_per_fixture(study, out / "per_fixture_data")
        print(f"Per-fixture CSV+PNG : {len(files)} files in "
             f"{out / 'per_fixture_data'}")

    arc = EvoArchive.open(args.archive)
    imgs = arc.export_images(out / "images")
    if imgs:
        print(f"Images     : {len(imgs)} recovered into {out / 'images'}")
    return 0


def cmd_export(args) -> int:
    study = load_study(args.archive, args.mapping, verbose=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fmts = {f.strip().lower() for f in args.format.split(",")}

    if "csv" in fmts:
        files = export_csv(study, out)
        print(f"CSV  : {len(files)} files")
    if "json" in fmts:
        print(f"JSON : {export_json(study, out / 'study.json')}")
    if "pdf" in fmts:
        print(f"PDF  : {export_pdf(study, out / 'study.pdf')}")
    if "png" in fmts:
        print(f"PNG  : {len(export_pngs(study, out / 'sheets'))} files")
    if "images" in fmts:
        arc = EvoArchive.open(args.archive)
        print(f"IMG  : {len(arc.export_images(out / 'images'))} files")
    if "dashboard" in fmts:
        print(f"HTML : {export_dashboard(study, out / 'dashboard.html')}")
    if "per-fixture" in fmts or "per_fixture" in fmts:
        files = export_per_fixture(study, out / "per_fixture_data")
        print(f"FIX  : {len(files)} per-fixture CSV+PNG files")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evostudy", description=BANNER)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("archive", help=".evo file or extracted folder")
        return sp

    s = common(sub.add_parser("inspect", help="list every file with its real type"))
    s.set_defaults(func=cmd_inspect)

    s = common(sub.add_parser("typeschema",
               help="list/inspect a PersistenceObject type registry "
                    "(for evo versions where ProjectData.xml is a schema, "
                    "not literal data)"))
    s.add_argument("--object", help="substring to filter object type names")
    s.set_defaults(func=cmd_typeschema)

    s = common(sub.add_parser("schema", help="dump the XML tag tree"))
    s.add_argument("--file", help="specific XML inside the archive")
    s.add_argument("--lines", type=int, default=250)
    s.set_defaults(func=cmd_schema)

    s = common(sub.add_parser("probe", help="analyse .rsl / .dat binaries"))
    s.add_argument("--file", help="specific file inside the archive")
    s.add_argument("--top", type=int, default=5)
    s.set_defaults(func=cmd_probe)

    s = common(sub.add_parser("dashboard",
               help="build a self-contained interactive HTML dashboard "
                    "(open in a browser — no install needed)"))
    s.add_argument("-o", "--out", default="evostudy_out")
    s.add_argument("--mapping", help="tag mapping JSON")
    s.add_argument("--per-fixture", action="store_true",
                   help="also export a CSV+PNG estimated map per luminaire")
    s.set_defaults(func=cmd_dashboard)

    s = common(sub.add_parser("render", help="draw the full study"))
    s.add_argument("-o", "--out", default="evostudy_out")
    s.add_argument("--mapping", help="tag mapping JSON")
    s.add_argument("--png", action="store_true", help="also write PNG sheets")
    s.add_argument("--dpi", type=int, default=200)
    s.add_argument("--no-3d", action="store_true")
    s.add_argument("--per-fixture", action="store_true",
                   help="also export a CSV+PNG estimated map per luminaire")
    s.set_defaults(func=cmd_render)

    s = common(sub.add_parser("export", help="export data (csv/json/pdf/png/images)"))
    s.add_argument("-o", "--out", default="evostudy_out")
    s.add_argument("--mapping", help="tag mapping JSON")
    s.add_argument("--format", default="csv,json")
    s.set_defaults(func=cmd_export)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
