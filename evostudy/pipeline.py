"""High-level pipeline: archive -> Study -> sheets / exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .archive import EvoArchive
from .model import CalcGrid, Study
from .parse_rsl import read_all_groups
from .parse_xml import ProjectParser, load_mapping


def load_study(path: str, mapping_file: Optional[str] = None,
               verbose: bool = False, progress=None) -> Study:
    """
    Open an .evo archive (or extracted folder) and recover everything.

    `progress`, if given, is called as `progress(message, frac)` with
    frac climbing 0..1 across real completed steps — parsing the XML,
    then each result group as it's actually read — not a fake animation.
    """
    def _tick(msg, frac):
        if progress is not None:
            progress(msg, frac)

    _tick("Opening archive", 0.0)
    arc = EvoArchive.open(path)

    proj = arc.project_xml
    if proj is None:
        raise FileNotFoundError(
            "Project/ProjectData/ProjectData.xml not found. "
            "Is this really a DIALux evo archive?")

    _tick("Parsing project data (rooms, luminaires)", 0.15)
    parser = ProjectParser(load_mapping(mapping_file))
    info = arc.archive_info.read() if arc.archive_info else None
    study = parser.parse(proj.read(), info)
    study.source_path = str(path)

    # ProjectData.xml is where this parser looks for luminaires/rooms — but
    # on evo builds where that file is only a type/schema registry (see
    # parse_xml.type_registry), the actual instance data lives in the
    # sibling ProjectData.dat instead, and this pipeline never opens it.
    # We don't guess its layout here (that needs real bytes to verify
    # against), but we make the blind spot visible instead of silently
    # reporting 0 luminaires as if ProjectData.xml were the whole story.
    if not study.luminaires and not study.rooms and arc.project_dat is not None:
        _tick("Recovering luminaires from ProjectData.dat", 0.20)
        from .parse_step import extract_luminaires
        try:
            dat_text = arc.project_dat.read().decode("utf-8", errors="replace")
            recovered, step_warnings = extract_luminaires(dat_text)
        except Exception as exc:
            recovered, step_warnings = [], [f"ProjectData.dat STEP recovery failed: {exc}"]
        if recovered:
            study.luminaires = recovered
            study.warnings.extend(step_warnings)
        else:
            study.warnings.append(
                f"ProjectData.xml matched no luminaires/rooms, and the sibling "
                f"{arc.project_dat.path} ({arc.project_dat.size:,} bytes) was "
                f"read as a STEP (ISO 10303-21) file but no LuminaireElement "
                f"records were found in it — this evo version may use a "
                f"different record type name. Run `evostudy probe --file "
                f"{arc.project_dat.path}` and inspect it directly.")

    # Results
    groups = arc.result_groups()
    if groups:
        extent = None
        if study.rooms and study.rooms[0].outline:
            extent = study.rooms[0].outline.bounds()
        elif study.geometry:
            extent = study.geometry[0].bounds()

        def _group_progress(done, total, gname):
            # Result-group recovery is the slow part on projects with many
            # groups, so it gets the bulk of the remaining range (0.25-1.0).
            frac = 0.25 + 0.75 * (done / total if total else 1.0)
            _tick(f"Recovering results ({done}/{total}): {gname}", frac)

        study.grids, study.environment_readings = read_all_groups(
            groups, room_aspect=study.aspect(), room_extent=extent,
            progress=_group_progress)
        if not study.grids:
            study.warnings.append(
                "Result groups were found but no illuminance array could be "
                "recovered from the .rsl files. Use `evostudy probe` to look "
                "at the candidates manually.")
        # Multiple quantity arrays per source file are now each rasterised
        # into their own grid (see parse_rsl.read_group), rather than only
        # the top-scoring one. Warn per SOURCE FILE (not per grid, which
        # would repeat the same message once per quantity already recovered
        # from that file) about any leads that scored too low to be used.
        warned_sources = set()
        for g in study.grids:
            if g.source in warned_sources or not g.other_candidates:
                continue
            still_unused = len(g.other_candidates) - sum(
                1 for gg in study.grids if gg.source == g.source)
            if still_unused > 0:
                warned_sources.add(g.source)
                study.warnings.append(
                    f"{g.source}: {sum(1 for gg in study.grids if gg.source == g.source)} "
                    f"quantity array(s) recovered and rasterised as separate "
                    f"grids; {still_unused} other candidate array(s) in the "
                    f"same file scored too low to use automatically — "
                    f"possibly noise, possibly another real quantity. See "
                    f"grid.other_candidates in study.json.")
    else:
        study.warnings.append(
            "No Project/Results groups in the archive — the project was saved "
            "without a completed calculation. Recalculate in DIALux and save.")
    _tick("Done reading project", 1.0)

    if verbose:
        print(f"archive: {len(arc)} files")
        print(f"luminaires: {len(study.luminaires)}")
        print(f"rooms: {len(study.rooms)}  polygons: {len(study.geometry)}")
        print(f"result grids: {len(study.grids)}")
        for w in study.warnings:
            print(f"  ! {w}")
    return study


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_dashboard(study: Study, out: str) -> str:
    """Write the self-contained interactive HTML dashboard."""
    from .render import build_dashboard
    out = str(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(build_dashboard(study), encoding="utf-8")
    return out


def export_per_fixture(study: Study, outdir: str, dpi: int = 150,
                       max_fixtures: int = 60) -> List[str]:
    """
    Write one CSV grid and one PNG map per luminaire, per calculation
    surface — the estimated direct-light footprint of that fixture alone.
    See evostudy.estimate for what "estimated" means here.
    """
    from matplotlib import pyplot as plt
    from .colors import band_norm, false_colour_cmap, nice_levels
    from .estimate import estimate_all_fixtures

    d = Path(outdir); d.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for gi, grid in enumerate(study.grids):
        if grid.values is None:
            continue
        contribs = estimate_all_fixtures(study, grid, max_fixtures=max_fixtures)
        gdir = d / f"surface_{gi+1}_{_slug(grid.name)}" / "per_fixture"
        gdir.mkdir(parents=True, exist_ok=True)

        for i, c in enumerate(contribs, start=1):
            base = f"fixture_{i:02d}_{_slug(c.label)}"

            # CSV
            x0, y0, x1, y1 = c.extent
            xs = np.linspace(x0, x1, c.values.shape[1])
            ys = np.linspace(y0, y1, c.values.shape[0])
            csv_path = gdir / f"{base}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["y\\x"] + [f"{x:.3f}" for x in xs])
                for j, yv in enumerate(ys):
                    w.writerow([f"{yv:.3f}"] +
                              [f"{v:.1f}" for v in c.values[j]])
            written.append(str(csv_path))

            # PNG
            m = c.metrics()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            levels = nice_levels(float(np.nanmin(c.values)),
                                 float(np.nanmax(c.values)) or 1.0, n=10)
            im = ax.imshow(c.values, origin="lower", extent=(x0, x1, y0, y1),
                           cmap=false_colour_cmap(len(levels) - 1),
                           norm=band_norm(levels), interpolation="bilinear")
            ax.plot(c.luminaire.x, c.luminaire.y, "o", color="#f5a623",
                   markeredgecolor="#7a4d00", markersize=7)
            ax.set_title(f"{c.label}  (ESTIMATED direct light)", fontsize=9)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            fig.colorbar(im, ax=ax, label="lx (estimated)")
            fig.text(0.5, 0.01, f"Eav {m.get('Eav',0):.1f}  Emax "
                    f"{m.get('Emax',0):.1f} lx — physics estimate, not "
                    f"DIALux output", ha="center", fontsize=6.5, color="#888")
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            png_path = gdir / f"{base}.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(str(png_path))

    return written


def export_pdf(study: Study, out: str, include_3d: bool = True) -> str:
    """Write the full multi-sheet report, DIALux style."""
    return export_report(study, pdf_path=out, include_3d=include_3d)[0]


def export_pngs(study: Study, outdir: str, dpi: int = 200) -> List[str]:
    return export_report(study, png_dir=outdir, dpi=dpi)[1]


def export_report(study: Study, pdf_path: Optional[str] = None,
                  png_dir: Optional[str] = None, dpi: int = 200,
                  include_3d: bool = True, progress=None):
    """
    Build every report sheet exactly once and write it to whichever of
    PDF / PNG-folder were asked for.

    Calling export_pdf() and export_pngs() back to back used to rebuild
    every sheet from scratch a second time — including the 3D surface
    plot, which is the slowest single figure Matplotlib produces here
    (per-facet depth sorting) — roughly doubling total render time for
    no reason. This renders each figure once and reuses it for both.
    `progress`, if given, is called as `progress(done, total, sheet_name)`
    after each sheet is drawn.
    """
    from matplotlib import pyplot as plt
    from .render import (
        sheet_false_colour, sheet_isolines, sheet_layout,
        sheet_summary, sheet_surface_3d, sheet_value_grid,
    )

    pdf_writer = None
    if pdf_path:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = str(pdf_path)
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        pdf_writer = PdfPages(pdf_path)
    if png_dir:
        png_dir = Path(png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    # (filename stem, figure-builder) for every sheet in the report
    jobs = [("00_summary", lambda: sheet_summary(study, study.grids)),
            ("01_layout", lambda: sheet_layout(study))]
    for gi, g in enumerate(study.grids, start=1):
        if g.values is None:
            continue
        base = f"{gi + 1:02d}_{_slug(g.name)}"
        jobs.append((f"{base}_falsecolour", lambda g=g: sheet_false_colour(study, g)))
        jobs.append((f"{base}_isolines", lambda g=g: sheet_isolines(study, g)))
        jobs.append((f"{base}_values", lambda g=g: sheet_value_grid(study, g)))
        if include_3d:
            jobs.append((f"{base}_3d", lambda g=g: sheet_surface_3d(study, g)))

    written_pngs: List[str] = []
    total = len(jobs)
    for i, (name, build) in enumerate(jobs):
        fig = build()
        if pdf_writer is not None:
            pdf_writer.savefig(fig)
        if png_dir is not None:
            p = png_dir / f"{name}.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            written_pngs.append(str(p))
        plt.close(fig)
        if progress is not None:
            progress(i + 1, total, name)

    if pdf_writer is not None:
        d = pdf_writer.infodict()
        d["Title"] = study.project_name or "Lighting study"
        d["Author"] = study.author or ""
        d["Subject"] = "Recovered from DIALux evo archive by evostudy"
        pdf_writer.close()

    return pdf_path, written_pngs


def export_csv(study: Study, outdir: str) -> List[str]:
    """Luminaire schedule + one CSV grid per calculation surface."""
    d = Path(outdir); d.mkdir(parents=True, exist_ok=True)
    written = []

    lum = d / "luminaires.csv"
    with lum.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "manufacturer", "article_no",
                    "x_m", "y_m", "z_m", "rotation_deg",
                    "flux_lm", "power_W", "efficacy_lm_W", "cct_K", "count"])
        for l in study.luminaires:
            w.writerow([l.id, l.name, l.manufacturer, l.article_no,
                        f"{l.x:.4f}", f"{l.y:.4f}", f"{l.z:.4f}",
                        f"{l.rotation:.2f}", l.luminous_flux or "",
                        l.power or "", f"{l.efficacy:.1f}" if l.efficacy else "",
                        l.cct or "", l.count])
    written.append(str(lum))

    for i, g in enumerate(study.grids, start=1):
        if g.values is None:
            continue
        p = d / f"grid_{i}_{_slug(g.name)}.csv"
        x0, y0, x1, y1 = g.extent or (0, 0, g.nx, g.ny)
        xs = np.linspace(x0, x1, g.nx)
        ys = np.linspace(y0, y1, g.ny)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["y\\x"] + [f"{x:.3f}" for x in xs])
            for j, yv in enumerate(ys):
                w.writerow([f"{yv:.3f}"] + [f"{v:.1f}" for v in g.values[j]])
        written.append(str(p))

    meta = d / "results_summary.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["surface", "Eav_lx", "Emin_lx", "Emax_lx",
                    "u0", "Emin/Emax", "points", "nx", "ny", "confidence"])
        for g in study.grids:
            m = g.metrics()
            if not m:
                continue
            w.writerow([g.name, f"{m['Eav']:.1f}", f"{m['Emin']:.1f}",
                        f"{m['Emax']:.1f}", f"{m['u0']:.3f}",
                        f"{m['Emin/Emax']:.3f}", m["points"],
                        g.nx, g.ny, f"{g.confidence:.2f}"])
    written.append(str(meta))
    return written


def export_json(study: Study, out: str) -> str:
    """Machine-readable dump of the whole recovered study."""
    def grid_dict(g: CalcGrid):
        return {
            "group": g.group, "name": g.name, "nx": g.nx, "ny": g.ny,
            "extent": g.extent, "height": g.height,
            "confidence": g.confidence, "source": g.source,
            "metrics": g.metrics(),
            "other_candidates": g.other_candidates,
            "values": g.values.round(2).tolist() if g.values is not None else None,
        }

    payload = {
        "project": {
            "name": study.project_name, "description": study.description,
            "author": study.author, "company": study.company,
            "dialux_version": study.dialux_version, "created": study.created,
            "source": study.source_path,
        },
        "totals": {
            "luminaire_count": len(study.luminaires),
            "total_flux_lm": study.total_flux(),
            "total_power_W": study.total_power(),
            "area_m2": study.total_area(),
            "power_density_W_m2": study.power_density(),
        },
        "luminaire_schedule": study.luminaire_schedule(),
        "luminaires": [
            {"id": l.id, "name": l.name, "manufacturer": l.manufacturer,
             "article_no": l.article_no, "x": l.x, "y": l.y, "z": l.z,
             "rotation": l.rotation, "flux_lm": l.luminous_flux,
             "power_W": l.power, "cct_K": l.cct, "count": l.count}
            for l in study.luminaires
        ],
        "rooms": [
            {"id": r.id, "name": r.name, "height": r.height,
             "area_m2": r.area,
             "outline": r.outline.points if r.outline else None}
            for r in study.rooms
        ],
        "grids": [grid_dict(g) for g in study.grids],
        "warnings": study.warnings,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return out


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return keep.strip("_")[:40] or "surface"
