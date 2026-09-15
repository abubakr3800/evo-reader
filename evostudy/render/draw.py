"""
Drawing routines that reproduce the DIALux evo output sheets:

  * Luminaire layout plan (with dimensions and a luminaire legend)
  * False-colour illuminance rendering
  * Isoline (iso-lux contour) plan
  * Value grid — the numeric lux value printed at each calculation point
  * Grey-scale 3-D surface of the illuminance distribution
  * Summary / result sheet with the standard metrics table
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must happen before pyplot is touched.
                        # Without this, on Windows/macOS with a GUI backend
                        # available (TkAgg etc.), creating a figure from any
                        # thread other than the main one can hang forever
                        # instead of raising — exactly what a Flask
                        # background-job worker does.
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle

from ..colors import (
    PLAN,
    band_norm,
    false_colour_cmap,
    nice_levels,
    value_text_colour,
)
from ..model import CalcGrid, Study

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

A4_LANDSCAPE = (11.69, 8.27)
A4_PORTRAIT = (8.27, 11.69)


# ---------------------------------------------------------------------------
# Shared furniture
# ---------------------------------------------------------------------------

def _sheet(title: str, subtitle: str = "", figsize=A4_LANDSCAPE):
    """Create a figure with the evo-style title block."""
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.08)
    fig.text(0.06, 0.955, title, fontsize=15, fontweight="bold",
             color=PLAN["text"])
    if subtitle:
        fig.text(0.06, 0.928, subtitle, fontsize=9, color="#555555")
    fig.add_artist(plt.Line2D([0.06, 0.95], [0.915, 0.915],
                              color=PLAN["accent"], lw=1.4,
                              transform=fig.transFigure))
    return fig


def _footer(fig, study: Study, page: str = "") -> None:
    left = " · ".join(x for x in (study.project_name, study.company,
                                  study.author) if x)
    fig.text(0.06, 0.035, left or "DIALux evo project", fontsize=7,
             color="#777777")
    fig.text(0.95, 0.035, page, fontsize=7, color="#777777", ha="right")
    fig.add_artist(plt.Line2D([0.06, 0.95], [0.055, 0.055],
                              color="#dddddd", lw=0.8,
                              transform=fig.transFigure))


def _draw_outlines(ax, study: Study, filled: bool = True) -> None:
    """
    Room walls and any other geometry, in plan.

    On result sheets pass filled=False, otherwise the room's floor colour
    paints over the illuminance rendering underneath it.
    """
    for room in study.rooms:
        if room.outline and len(room.outline.points) >= 3:
            ax.add_patch(MplPolygon(
                room.outline.points, closed=True,
                facecolor=PLAN["room_fill"] if filled else "none",
                edgecolor=PLAN["wall"],
                lw=PLAN["wall_lw"], zorder=2 if filled else 5))
    for poly in study.geometry:
        if len(poly.points) >= 3:
            ax.add_patch(MplPolygon(
                poly.points, closed=True, facecolor="none",
                edgecolor=PLAN["wall"], lw=1.0, zorder=3, alpha=0.8))


def _draw_luminaires(ax, study: Study, size: float = 0.30,
                     label: bool = False) -> None:
    for i, l in enumerate(study.luminaires, start=1):
        ax.add_patch(Rectangle(
            (l.x - size / 2, l.y - size / 2), size, size,
            angle=l.rotation,
            facecolor=PLAN["luminaire"], edgecolor=PLAN["luminaire_edge"],
            lw=0.7, zorder=8))
        # Little emission cross so orientation reads at a glance
        ax.plot([l.x - size * 0.75, l.x + size * 0.75], [l.y, l.y],
                color=PLAN["luminaire_edge"], lw=0.5, zorder=9)
        if label:
            ax.annotate(str(i), (l.x, l.y), fontsize=5.5, ha="center",
                        va="center", zorder=10, color="#3a2400")


def _apply_extent(ax, study: Study, grid: Optional[CalcGrid] = None) -> None:
    """
    Frame the drawing so that everything is visible. The calculation grid and
    the building geometry do not always cover the same area, so the view is
    the union of both rather than whichever one was drawn last.
    """
    bx0, by0, bx1, by1 = study.bounds()
    if grid is not None and grid.extent:
        gx0, gy0, gx1, gy1 = grid.extent
        pad = max(gx1 - gx0, gy1 - gy0) * 0.06 + 0.25
        x0, y0 = min(bx0, gx0 - pad), min(by0, gy0 - pad)
        x1, y1 = max(bx1, gx1 + pad), max(by1, gy1 + pad)
    else:
        x0, y0, x1, y1 = bx0, by0, bx1, by1
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, color=PLAN["grid"], lw=0.4, zorder=0)
    for s in ax.spines.values():
        s.set_color("#999999")


def _metrics_box(ax, grid: CalcGrid, loc=(1.02, 1.0), fig=None,
                 rect=None) -> None:
    """
    Render the standard result table. Pass `fig` and `rect` to place it in its
    own region of the sheet, which avoids colliding with a colour bar.
    """
    m = grid.metrics()
    if not m:
        return
    rows = [
        ("Eav",        f"{m['Eav']:.0f} lx"),
        ("Emin",       f"{m['Emin']:.0f} lx"),
        ("Emax",       f"{m['Emax']:.0f} lx"),
        ("u0 (Emin/Eav)",  f"{m['u0']:.2f}"),
        ("Emin/Emax",  f"{m['Emin/Emax']:.2f}"),
        ("Emax/Eav",   f"{m['Emax/Eav']:.2f}"),
        ("Grid",       f"{grid.nx} x {grid.ny}"),
    ]
    txt = "\n".join(f"{k:<16}{v:>10}" for k, v in rows)
    if fig is not None and rect is not None:
        tax = fig.add_axes(rect)
        tax.axis("off")
        tax.text(0, 1, txt, fontsize=7.2, family="DejaVu Sans Mono",
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f4",
                           edgecolor="#cccccc", lw=0.7))
        return
    ax.text(loc[0], loc[1], txt, transform=ax.transAxes, fontsize=7.2,
            family="DejaVu Sans Mono", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f4",
                      edgecolor="#cccccc", lw=0.7))


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sheet_layout(study: Study):
    """Luminaire layout plan + schedule table."""
    fig = _sheet("Luminaire layout plan",
                 f"{study.project_name} — plan view, dimensions in metres")
    ax = fig.add_axes([0.06, 0.30, 0.56, 0.58])
    _draw_outlines(ax, study)
    _draw_luminaires(ax, study, label=True)
    _apply_extent(ax, study)

    # Schedule
    rows = study.luminaire_schedule()
    tax = fig.add_axes([0.66, 0.30, 0.29, 0.58]); tax.axis("off")
    tax.text(0, 1.0, "Luminaire schedule", fontsize=9.5, fontweight="bold",
             va="top")
    y = 0.94
    if rows:
        hdr = f"{'#':<4}{'Luminaire':<26}{'Qty':>5}{'Φ [lm]':>10}{'P [W]':>8}"
        tax.text(0, y, hdr, fontsize=6.8, family="DejaVu Sans Mono",
                 va="top", fontweight="bold")
        y -= 0.035
        for i, r in enumerate(rows[:22], start=1):
            nm = (r["name"] or "")[:25]
            flux = f"{r['flux']:.0f}" if r["flux"] else "—"
            pw = f"{r['power']:.1f}" if r["power"] else "—"
            tax.text(0, y,
                     f"{i:<4}{nm:<26}{r['count']:>5}{flux:>10}{pw:>8}",
                     fontsize=6.5, family="DejaVu Sans Mono", va="top")
            y -= 0.030
    else:
        tax.text(0, y, "No luminaires recovered from ProjectData.xml.\n"
                       "Run `evostudy schema` and tune the tag mapping.",
                 fontsize=7.5, va="top", color="#a33")
        y -= 0.08

    # Totals
    y -= 0.02
    tot = []
    if study.total_power():
        tot.append(("Total connected load", f"{study.total_power():.1f} W"))
    if study.total_flux():
        tot.append(("Total luminous flux", f"{study.total_flux():.0f} lm"))
    if study.total_area():
        tot.append(("Area", f"{study.total_area():.2f} m²"))
    if study.power_density():
        tot.append(("Specific load", f"{study.power_density():.2f} W/m²"))
    if study.power_density_per_100lx():
        tot.append(("per 100 lx",
                    f"{study.power_density_per_100lx():.2f} W/m²/100lx"))
    if tot:
        tax.text(0, y, "Totals", fontsize=9, fontweight="bold", va="top")
        y -= 0.035
        for k, v in tot:
            tax.text(0, y, f"{k:<24}{v:>16}", fontsize=6.8,
                     family="DejaVu Sans Mono", va="top")
            y -= 0.030

    _footer(fig, study, "Layout")
    return fig


def sheet_false_colour(study: Study, grid: CalcGrid):
    """False-colour illuminance rendering with a banded legend."""
    fig = _sheet("Illuminance — false colour rendering",
                 f"{grid.name} · working plane"
                 + (f" at {grid.height:.2f} m" if grid.height is not None else ""))
    ax = fig.add_axes([0.05, 0.12, 0.54, 0.74])

    v = grid.values
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    levels = nice_levels(vmin, vmax, n=12)
    cmap = false_colour_cmap(len(levels) - 1)
    norm = band_norm(levels)

    x0, y0, x1, y1 = grid.extent or (0, 0, grid.nx, grid.ny)
    im = ax.imshow(v, origin="lower", extent=(x0, x1, y0, y1),
                   cmap=cmap, norm=norm, interpolation="bilinear", zorder=1)

    _draw_outlines(ax, study, filled=False)
    _draw_luminaires(ax, study, size=0.22)
    _apply_extent(ax, study, grid)
    ax.grid(False)

    cax = fig.add_axes([0.625, 0.12, 0.020, 0.74])
    cb = fig.colorbar(im, cax=cax, ticks=levels, spacing="uniform")
    cb.ax.tick_params(labelsize=6.2)
    cb.ax.set_yticklabels([f"{l:g}" for l in levels])
    cb.set_label("Illuminance E [lx]", fontsize=7.5)

    _metrics_box(ax, grid, fig=fig, rect=[0.72, 0.60, 0.26, 0.26])
    _footer(fig, study, "False colour")
    return fig


def sheet_isolines(study: Study, grid: CalcGrid):
    """Iso-lux contour plan — the classic DIALux isoline sheet."""
    fig = _sheet("Illuminance — isolines",
                 f"{grid.name} · values in lx")
    ax = fig.add_axes([0.05, 0.12, 0.62, 0.74])

    v = grid.values
    x0, y0, x1, y1 = grid.extent or (0, 0, grid.nx, grid.ny)
    xs = np.linspace(x0, x1, grid.nx)
    ys = np.linspace(y0, y1, grid.ny)
    X, Y = np.meshgrid(xs, ys)

    levels = nice_levels(float(np.nanmin(v)), float(np.nanmax(v)), n=9)
    levels = sorted(set(round(l, 2) for l in levels))

    ax.contourf(X, Y, v, levels=levels, cmap=false_colour_cmap(len(levels) - 1),
                alpha=0.30, zorder=1)
    cs = ax.contour(X, Y, v, levels=levels, colors="#33383d",
                    linewidths=0.8, zorder=4)
    ax.clabel(cs, inline=True, fontsize=6, fmt=lambda x: f"{x:.0f}")

    _draw_outlines(ax, study, filled=False)
    _draw_luminaires(ax, study, size=0.22)
    _apply_extent(ax, study, grid)

    _metrics_box(ax, grid, fig=fig, rect=[0.72, 0.60, 0.26, 0.26])
    _footer(fig, study, "Isolines")
    return fig


def sheet_value_grid(study: Study, grid: CalcGrid, max_labels: int = 400):
    """The numeric value at each calculation point, as evo prints it."""
    fig = _sheet("Illuminance — calculation point values",
                 f"{grid.name} · E in lx at each grid point")
    ax = fig.add_axes([0.05, 0.12, 0.62, 0.74])

    v = grid.values
    x0, y0, x1, y1 = grid.extent or (0, 0, grid.nx, grid.ny)
    xs = np.linspace(x0, x1, grid.nx)
    ys = np.linspace(y0, y1, grid.ny)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))

    levels = nice_levels(vmin, vmax, n=10)
    ax.imshow(v, origin="lower", extent=(x0, x1, y0, y1),
              cmap=false_colour_cmap(len(levels) - 1), norm=band_norm(levels),
              interpolation="nearest", alpha=0.55, zorder=1)

    # Thin the labels. A 48x32 grid is 1536 numbers, which is unreadable on a
    # sheet, so print roughly a 16x12 sample the way DIALux does.
    import matplotlib.patheffects as pe

    max_cols, max_rows = 16, 12
    step_x = max(1, int(np.ceil(grid.nx / max_cols)))
    step_y = max(1, int(np.ceil(grid.ny / max_rows)))

    for j in range(step_y // 2, grid.ny, step_y):
        for i in range(step_x // 2, grid.nx, step_x):
            val = v[j, i]
            if not np.isfinite(val):
                continue
            ax.text(xs[i], ys[j], f"{val:.0f}", fontsize=6.0, ha="center",
                    va="center", zorder=6, color="#15181c",
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground="white")])

    _draw_outlines(ax, study, filled=False)
    _draw_luminaires(ax, study, size=0.18)
    _apply_extent(ax, study, grid)
    ax.grid(False)

    _metrics_box(ax, grid, fig=fig, rect=[0.72, 0.60, 0.26, 0.26])
    _footer(fig, study, "Values")
    return fig


def sheet_surface_3d(study: Study, grid: CalcGrid):
    """3-D illuminance surface — evo's 'grey-scale / 3D' result view."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = _sheet("Illuminance distribution — 3D surface", grid.name)
    ax = fig.add_axes([0.06, 0.08, 0.86, 0.78], projection="3d")

    v = grid.values
    x0, y0, x1, y1 = grid.extent or (0, 0, grid.nx, grid.ny)
    X, Y = np.meshgrid(np.linspace(x0, x1, grid.nx),
                       np.linspace(y0, y1, grid.ny))
    levels = nice_levels(float(np.nanmin(v)), float(np.nanmax(v)), n=12)
    ax.plot_surface(X, Y, v, cmap=false_colour_cmap(len(levels) - 1),
                    norm=band_norm(levels), linewidth=0, antialiased=True,
                    rstride=max(1, grid.ny // 60), cstride=max(1, grid.nx // 60))
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.set_zlabel("E [lx]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=38, azim=-125)
    _footer(fig, study, "3D")
    return fig


def sheet_summary(study: Study, grids: List[CalcGrid]):
    """Cover / summary sheet with project data and the headline results."""
    fig = _sheet("Lighting study — summary",
                 study.project_name, figsize=A4_PORTRAIT)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.80]); ax.axis("off")

    y = 1.0
    def block(title, rows, y):
        ax.text(0, y, title, fontsize=10.5, fontweight="bold", va="top")
        y -= 0.028
        for k, v in rows:
            ax.text(0.0, y, f"{k}", fontsize=8, va="top", color="#444444")
            ax.text(0.42, y, f"{v}", fontsize=8, va="top",
                    family="DejaVu Sans Mono")
            y -= 0.022
        return y - 0.018

    y = block("Project", [
        ("Name", study.project_name or "—"),
        ("Description", study.description or "—"),
        ("Author", study.author or "—"),
        ("Company", study.company or "—"),
        ("DIALux version", study.dialux_version or "—"),
        ("Created", study.created or "—"),
        ("Source archive", study.source_path or "—"),
    ], y)

    y = block("Installation", [
        ("Luminaires placed", str(len(study.luminaires))),
        ("Distinct types", str(len(study.luminaire_schedule()))),
        ("Total luminous flux",
         f"{study.total_flux():.0f} lm" if study.total_flux() else "—"),
        ("Total connected load",
         f"{study.total_power():.1f} W" if study.total_power() else "—"),
        ("Area",
         f"{study.total_area():.2f} m²" if study.total_area() else "—"),
        ("Specific connected load",
         f"{study.power_density():.2f} W/m²" if study.power_density() else "—"),
    ], y)

    for g in grids:
        m = g.metrics()
        if not m:
            continue
        y = block(f"Results — {g.name}", [
            ("Average illuminance Eav", f"{m['Eav']:.0f} lx"),
            ("Minimum Emin", f"{m['Emin']:.0f} lx"),
            ("Maximum Emax", f"{m['Emax']:.0f} lx"),
            ("Uniformity u0 (Emin/Eav)", f"{m['u0']:.2f}"),
            ("Diversity Emin/Emax", f"{m['Emin/Emax']:.2f}"),
            ("Calculation points", f"{m['points']} ({g.nx} x {g.ny})"),
            ("Recovery confidence", f"{g.confidence:.0%}"),
        ], y)

    if study.warnings:
        ax.text(0, y, "Notes", fontsize=10.5, fontweight="bold", va="top")
        y -= 0.028
        for w in study.warnings[:8]:
            ax.text(0, y, f"• {w}", fontsize=7, va="top", color="#a33",
                    wrap=True)
            y -= 0.030

    _footer(fig, study, "Summary")
    return fig
