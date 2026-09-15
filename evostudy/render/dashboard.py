"""
A single self-contained HTML file that a normal user can double-click and
open in any browser — no Python, no install, no internet connection needed.

Contains:
  * A plain-language project summary
  * The luminaire schedule and room list, as normal tables
  * An interactive canvas heatmap per calculation surface, with:
      - a dropdown to switch between the combined (recovered) result and
        each individual fixture's ESTIMATED direct-light contribution
      - mouse hover showing the exact lux value under the pointer
      - click-to-toggle luminaire markers
  * Environment (wall/ceiling/floor) readings, if any were recoverable
  * A plain-language "how sure is this" section reporting confidence

No external JS/CSS libraries are used (no CDN), so it works fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..estimate import FixtureContribution, estimate_all_fixtures
from ..model import CalcGrid, Study
from .nav import PAGE_NAV_CSS, page_nav_html

CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#1b1f24; --sub:#6b7280;
  --accent:#c8102e; --border:#e3e5e9; --good:#1f9d55; --warn:#b8860b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
header{background:var(--panel);border-bottom:3px solid var(--accent);
  padding:20px 28px;}
header h1{margin:0 0 4px;font-size:22px}
header .sub{color:var(--sub);font-size:13px}
main{max-width:1200px;margin:0 auto;padding:24px 20px 60px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:14px;margin-bottom:26px}
.card{background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:16px 18px}
.card .label{font-size:12px;color:var(--sub);text-transform:uppercase;
  letter-spacing:.04em;margin-bottom:6px}
.card .value{font-size:22px;font-weight:600}
.card .value small{font-size:13px;font-weight:400;color:var(--sub)}
section{background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:20px 22px;margin-bottom:22px}
section h2{margin-top:0;font-size:16px;border-bottom:1px solid var(--border);
  padding-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border)}
th{color:var(--sub);font-weight:600;font-size:11px;text-transform:uppercase}
tr:hover td{background:#fafafa}
.badge{display:inline-block;padding:2px 9px;border-radius:100px;
  font-size:11px;font-weight:600}
.badge.good{background:#e7f7ee;color:var(--good)}
.badge.warn{background:#fdf4e3;color:var(--warn)}
.viewer{display:flex;gap:22px;flex-wrap:wrap}
.viewer .stage{flex:1 1 560px}
.viewer .side{flex:0 0 260px}
select{width:100%;padding:8px 10px;border:1px solid var(--border);
  border-radius:8px;font-size:13px;margin-bottom:12px;background:white}
canvas{width:100%;border:1px solid var(--border);border-radius:8px;
  cursor:crosshair;background:#fff}
.readout{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;
  background:#f7f7f8;border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;line-height:1.7}
.readout b{display:inline-block;width:120px;color:var(--sub);font-weight:500}
.hint{color:var(--sub);font-size:12.5px;margin-top:8px}
.legend{display:flex;align-items:center;gap:10px;margin-top:10px;
  font-size:11.5px;color:var(--sub)}
.legend .bar{flex:1;height:14px;border-radius:4px;
  background:linear-gradient(to right,#1b1464,#2b3a9c,#2f6fd0,#2bb1d6,
  #2ecfa2,#7ad64f,#c9e04a,#f5e04a,#f7b93e,#f28a2e,#e85c2a,#d63333,#f8b6c8)}
.note{font-size:12.5px;color:var(--sub);background:#fdf4e3;
  border:1px solid #f0dfb0;border-radius:8px;padding:10px 12px;margin-top:10px}
footer{text-align:center;color:var(--sub);font-size:12px;padding:30px 0 10px}
""" + PAGE_NAV_CSS

JS_HEATMAP = r"""
function drawHeatmap(canvas, values, nx, ny, extent, luminaires, label) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 560;
  const cssH = Math.round(cssW * (ny / nx)) || 400;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  let vmin = Infinity, vmax = -Infinity;
  for (const row of values) for (const v of row) {
    if (v < vmin) vmin = v; if (v > vmax) vmax = v;
  }
  if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
  if (vmax <= vmin) vmax = vmin + 1;

  const stops = [
    [27,20,100],[43,58,156],[47,111,208],[43,177,214],[46,207,162],
    [122,214,79],[201,224,74],[245,224,74],[247,185,62],[242,138,46],
    [232,92,42],[214,51,51],[248,182,200]
  ];
  function colour(t) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (stops.length - 1);
    const i = Math.floor(f), frac = f - i;
    const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
    const r = a[0] + (b[0]-a[0])*frac, g = a[1] + (b[1]-a[1])*frac,
          bl = a[2] + (b[2]-a[2])*frac;
    return `rgb(${r|0},${g|0},${bl|0})`;
  }

  const cw = cssW / nx, ch = cssH / ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = values[j][i];
      ctx.fillStyle = isFinite(v) ? colour((v - vmin) / (vmax - vmin)) : "#eee";
      // row 0 of `values` is the lowest y — flip so it draws at the bottom
      ctx.fillRect(i * cw, (ny - 1 - j) * ch, cw + 1, ch + 1);
    }
  }

  // luminaire markers
  const [x0, y0, x1, y1] = extent;
  ctx.fillStyle = "#f5a623"; ctx.strokeStyle = "#7a4d00"; ctx.lineWidth = 1.2;
  for (const l of luminaires) {
    const px = (l.x - x0) / (x1 - x0) * cssW;
    const py = cssH - (l.y - y0) / (y1 - y0) * cssH;
    ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();
  }

  canvas._meta = { vmin, vmax, nx, ny, extent, values, label };
}

function attachHover(canvas, readoutEl) {
  canvas.addEventListener("mousemove", (ev) => {
    const meta = canvas._meta; if (!meta) return;
    const rect = canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
    const [x0, y0, x1, y1] = meta.extent;
    const fx = cx / rect.width, fy = 1 - (cy / rect.height);
    const gx = x0 + fx * (x1 - x0), gy = y0 + fy * (y1 - y0);
    const i = Math.min(meta.nx - 1, Math.max(0, Math.round(fx * (meta.nx - 1))));
    const j = Math.min(meta.ny - 1, Math.max(0, Math.round(fy * (meta.ny - 1))));
    const v = meta.values[j][i];
    readoutEl.innerHTML =
      `<b>Position</b> ${gx.toFixed(2)} m, ${gy.toFixed(2)} m<br>` +
      `<b>Value</b> ${isFinite(v) ? v.toFixed(1) + " lx" : "—"}<br>` +
      `<b>Range shown</b> ${meta.vmin.toFixed(0)}\u2013${meta.vmax.toFixed(0)} lx`;
  });
  canvas.addEventListener("mouseleave", () => {
    const meta = canvas._meta; if (!meta) return;
    readoutEl.innerHTML =
      `<b>Eav</b> hover the map<br><b>Range shown</b> ` +
      `${meta.vmin.toFixed(0)}\u2013${meta.vmax.toFixed(0)} lx`;
  });
}

function initViewer(viewerId, data) {
  const root = document.getElementById(viewerId);
  const select = root.querySelector("select");
  const canvas = root.querySelector("canvas");
  const readout = root.querySelector(".readout");
  const legendMin = root.querySelector(".legend-min");
  const legendMax = root.querySelector(".legend-max");

  function render() {
    const key = select.value;
    const layer = data.layers[key];
    drawHeatmap(canvas, layer.values, data.nx, data.ny, data.extent,
                key === "combined" ? data.luminaires : [layer.luminaire],
                layer.label);
    legendMin.textContent = canvas._meta.vmin.toFixed(0) + " lx";
    legendMax.textContent = canvas._meta.vmax.toFixed(0) + " lx";
    readout.innerHTML =
      `<b>Showing</b> ${layer.label}<br>` +
      `<b>Eav</b> ${layer.Eav.toFixed(1)} lx &nbsp; ` +
      `<b>Emin</b> ${layer.Emin.toFixed(1)} &nbsp; ` +
      `<b>Emax</b> ${layer.Emax.toFixed(1)}<br>` +
      `<b>Range shown</b> ${canvas._meta.vmin.toFixed(0)}` +
      `\u2013${canvas._meta.vmax.toFixed(0)} lx`;
  }
  select.addEventListener("change", render);
  attachHover(canvas, readout);
  window.addEventListener("resize", render);
  render();
}
"""


MAX_DISPLAY_POINTS = 10_000  # cap per layer so embedded JSON stays browsable


def _downsample(values: np.ndarray, max_points: int = MAX_DISPLAY_POINTS
                ) -> np.ndarray:
    """
    Shrink a grid for on-screen display only (never touches the values used
    for CSV/JSON export or the printed metrics). Without this, a study with
    a few hundred fixtures and a large calc grid embeds one full-resolution
    copy of the grid PER fixture layer into the HTML — that is what made
    some studies' dashboard.html balloon to tens of megabytes and hang the
    browser. A plain stride-based subsample is enough for a heatmap preview.
    """
    ny, nx = values.shape
    if nx * ny <= max_points:
        return values
    factor = int(np.ceil(((nx * ny) / max_points) ** 0.5))
    return values[::factor, ::factor]


def _fixture_layers(study: Study, grid: CalcGrid,
                    contribs: List[FixtureContribution]) -> dict:
    disp = _downsample(grid.values)
    layers = {
        "combined": {
            "label": "Combined result (recovered from DIALux, all fixtures + interreflection)",
            "values": np.round(disp, 1).tolist(),
            **{k: v for k, v in grid.metrics().items() if k in
               ("Eav", "Emin", "Emax")},
        }
    }
    for i, c in enumerate(contribs):
        m = c.metrics()
        if not m:
            continue
        layers[f"fixture_{i}"] = {
            "label": f"Estimated direct light — {c.label} "
                    f"(#{i+1} at {c.luminaire.x:.2f}, {c.luminaire.y:.2f} m)",
            "values": np.round(_downsample(c.values), 1).tolist(),
            "Eav": m["Eav"], "Emin": m["Emin"], "Emax": m["Emax"],
        }
    return layers, disp.shape  # (ny, nx) actually displayed, after downsampling


def _viewer_html(viewer_id: str, title: str, study: Study, grid: CalcGrid,
                 contribs: List[FixtureContribution]) -> str:
    layers, (disp_ny, disp_nx) = _fixture_layers(study, grid, contribs)
    data = {
        "nx": disp_nx, "ny": disp_ny, "extent": list(grid.extent or (0, 0, grid.nx, grid.ny)),
        "luminaires": [{"x": l.x, "y": l.y} for l in study.luminaires],
        "layers": layers,
    }
    for key, c in zip([k for k in layers if k != "combined"], contribs):
        layers[key]["luminaire"] = {"x": c.luminaire.x, "y": c.luminaire.y}

    options = "\n".join(
        f'<option value="{k}">{v["label"]}</option>' for k, v in layers.items())

    return f"""
<section>
  <h2>{title}</h2>
  <div id="{viewer_id}" class="viewer">
    <div class="stage">
      <canvas></canvas>
      <div class="legend">
        <span class="legend-min">—</span>
        <div class="bar"></div>
        <span class="legend-max">—</span>
      </div>
      <div class="hint">Hover the map for the exact value at any point. The
      fixture dropdown shows an <b>estimated</b> direct-light footprint for
      one luminaire at a time — DIALux does not export true per-fixture
      results, so this is calculated here from that fixture's recovered
      position and flux, ignoring its real photometric distribution and any
      light bouncing off walls.</div>
    </div>
    <div class="side">
      <select>{options}</select>
      <div class="readout">Loading…</div>
    </div>
  </div>
  <script>
    initViewer("{viewer_id}", {json.dumps(data)});
  </script>
</section>
"""


def _confidence_badge(score: float) -> str:
    if score >= 0.75:
        return '<span class="badge good">high confidence</span>'
    if score >= 0.4:
        return '<span class="badge warn">check against DIALux</span>'
    return '<span class="badge warn">low confidence — verify</span>'


def build_dashboard(study: Study, max_fixture_layers: int = 10) -> str:
    """Render the full self-contained dashboard HTML for `study`."""
    rows_lum = ""
    for i, r in enumerate(study.luminaire_schedule()):
        flux = f"{r['flux']:.0f} lm" if r["flux"] else "—"
        power = f"{r['power']:.1f} W" if r["power"] else "—"
        rows_lum += (f"<tr><td>{i+1}</td><td>{r['name']}</td>"
                    f"<td>{r['manufacturer'] or '—'}</td>"
                    f"<td>{r['count']}</td><td>{flux}</td>"
                    f"<td>{power}</td></tr>")

    rows_rooms = ""
    for r in study.rooms:
        rows_rooms += (f"<tr><td>{r.name}</td>"
                       f"<td>{r.area:.2f} m²</td>"
                       f"<td>{r.height if r.height else '—'}</td>"
                       f"<td>{r.maintenance_factor if r.maintenance_factor else '—'}</td></tr>")
    if not rows_rooms:
        rows_rooms = ('<tr><td colspan="4">No room outline recovered — '
                      'geometry may need a tag-mapping adjustment '
                      '(see `evostudy schema`).</td></tr>')

    cards = f"""
    <div class="grid">
      <div class="card"><div class="label">Luminaires</div>
        <div class="value">{len(study.luminaires)}</div></div>
      <div class="card"><div class="label">Total flux</div>
        <div class="value">{study.total_flux():.0f} <small>lm</small></div></div>
      <div class="card"><div class="label">Connected load</div>
        <div class="value">{study.total_power():.0f} <small>W</small></div></div>
      <div class="card"><div class="label">Area</div>
        <div class="value">{study.total_area():.1f} <small>m²</small></div></div>
      <div class="card"><div class="label">Specific load</div>
        <div class="value">{(study.power_density() or 0):.2f} <small>W/m²</small></div></div>
    </div>
    """ if study.total_flux() and study.total_power() else "<div class='grid'></div>"

    viewers = ""
    for gi, grid in enumerate(study.grids):
        if grid.values is None:
            continue
        contribs = estimate_all_fixtures(study, grid, max_fixtures=max_fixture_layers)
        conf = _confidence_badge(grid.confidence)
        title = f"{grid.name or f'Surface {gi+1}'} — interactive illuminance map {conf}"
        viewers += _viewer_html(f"viewer{gi}", title, study, grid, contribs)

    env_html = ""
    if study.environment_readings:
        rows = "".join(
            f"<tr><td>{e.group}</td><td>{', '.join(f'{v:.1f}' for v in e.values[:12])}"
            f"{' …' if len(e.values) > 12 else ''}</td></tr>"
            for e in study.environment_readings)
        env_html = f"""
        <section>
          <h2>Environment surfaces — wall / ceiling / floor readings</h2>
          <table><tr><th>Result group</th><th>Recovered values (lx)</th></tr>
          {rows}</table>
          <div class="note">These are DIALux's reflected-light bookkeeping
          values for the room surfaces, not a full grid — there usually
          aren't enough of them to draw a map. Treat them as a rough
          indication, not an exact per-surface reading.</div>
        </section>
        """
    else:
        env_html = """
        <section>
          <h2>Environment surfaces — wall / ceiling / floor readings</h2>
          <p class="hint">No environment (indirect/surface) readings could be
          recovered from this archive's <code>Dataenvillum*.rsl</code> files.
          This can mean the project has no saved surface-illumination data,
          or that this evo version stores it differently than expected —
          run <code>evostudy probe</code> on that file to check by hand.</p>
        </section>
        """

    warn_html = ""
    if study.warnings:
        items = "".join(f"<li>{w}</li>" for w in study.warnings)
        warn_html = f"""
        <section>
          <h2>Notes on this recovery</h2>
          <ul>{items}</ul>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{study.project_name or "Lighting study"} — evostudy dashboard</title>
<style>{CSS}</style>
<script>{JS_HEATMAP}</script>
</head>
<body>
<header>
  <h1>{study.project_name or "Lighting study"}</h1>
  <div class="sub">{study.company or ""}{" · " if study.company and study.author else ""}{study.author or ""}
  {" · " if (study.company or study.author) and study.dialux_version else ""}{study.dialux_version or ""}</div>
</header>
{page_nav_html("dashboard.html")}
<main>
  {cards}

  <section>
    <h2>Luminaire schedule</h2>
    <table>
      <tr><th>#</th><th>Luminaire</th><th>Manufacturer</th><th>Qty</th>
          <th>Flux</th><th>Power</th></tr>
      {rows_lum or '<tr><td colspan="6">No luminaires recovered.</td></tr>'}
    </table>
  </section>

  <section>
    <h2>Rooms</h2>
    <table>
      <tr><th>Room</th><th>Area</th><th>Height</th><th>Maintenance factor</th></tr>
      {rows_rooms}
    </table>
  </section>

  {viewers}
  {env_html}
  {warn_html}
</main>
<footer>
  Generated by evostudy from {study.source_path or "the archive"}. Result
  maps are statistically recovered from DIALux's undocumented .rsl files;
  per-fixture views are physics estimates, not DIALux output. Always
  cross-check headline numbers (Eav/Emin/Emax) against DIALux itself.
</footer>
</body>
</html>"""
