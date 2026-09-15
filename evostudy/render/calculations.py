"""
The "Calculations" page — a second self-contained HTML export, next to
dashboard.html.

Where the dashboard's heatmap is a smooth, continuous preview you hover to
read a value, this page reproduces the way DIALux itself prints a result
sheet: the working plane split into the same discrete false-colour bands
DIALux uses (sourced from evostudy.colors, the identical palette the
PDF/PNG report sheets use — not a separate approximation), with the actual
lux number printed on top of each sampled cell. The right-hand panel lists
the numbers DIALux would print alongside that sheet: Eav/Emin/Emax, the two
uniformities, and where the darkest and brightest points actually are.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

from ..colors import band_hex_colours, nice_levels
from ..model import CalcGrid, Study
from .dashboard import _downsample
from .nav import PAGE_NAV_CSS, page_nav_html

CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#1b1f24; --sub:#6b7280;
  --accent:#c8102e; --border:#e3e5e9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
header{background:var(--panel);border-bottom:3px solid var(--accent);
  padding:20px 28px;}
header h1{margin:0 0 4px;font-size:22px}
header .sub{color:var(--sub);font-size:13px}
main{max-width:1280px;margin:0 auto;padding:24px 20px 60px}
section{background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:20px 22px;margin-bottom:22px}
section h2{margin-top:0;font-size:16px;border-bottom:1px solid var(--border);
  padding-bottom:10px}
.calc{display:flex;gap:22px;flex-wrap:wrap}
.calc .stage{flex:1 1 620px}
.calc .side{flex:0 0 300px}
canvas{width:100%;border:1px solid var(--border);border-radius:8px;background:#fff}
.legend-strip{display:flex;margin-top:10px;border-radius:6px;overflow:hidden;
  border:1px solid var(--border)}
.legend-strip div{flex:1;height:16px}
.legend-labels{display:flex;justify-content:space-between;font-size:10.5px;
  color:var(--sub);margin-top:3px;font-family:ui-monospace,Consolas,monospace}
table.metrics{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:14px}
table.metrics td{padding:6px 4px;border-bottom:1px solid var(--border)}
table.metrics td:first-child{color:var(--sub)}
table.metrics td:last-child{text-align:right;font-weight:600;
  font-family:ui-monospace,Consolas,monospace}
.pointbox{background:#f7f7f8;border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;font-size:12.5px;margin-bottom:10px}
.pointbox b{display:block;font-size:11px;text-transform:uppercase;
  letter-spacing:.04em;color:var(--sub);margin-bottom:4px}
.pointbox .dot{display:inline-block;width:9px;height:9px;border-radius:50%;
  margin-right:6px;vertical-align:1px}
.hint{color:var(--sub);font-size:12.5px;margin-top:8px}
footer{text-align:center;color:var(--sub);font-size:12px;padding:30px 0 10px}
""" + PAGE_NAV_CSS

JS = r"""
function drawValueChart(canvas, values, nx, ny, extent, levels, bandColours,
                        labelStepX, labelStepY, extrema) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 620;
  const cssH = Math.round(cssW * (ny / nx)) || 440;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  function bandOf(v) {
    let idx = 0;
    for (let k = 1; k < levels.length; k++) { if (v >= levels[k - 1]) idx = k - 1; }
    return Math.min(Math.max(idx, 0), bandColours.length - 1);
  }

  const cw = cssW / nx, ch = cssH / ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = values[j][i];
      ctx.fillStyle = isFinite(v) ? bandColours[bandOf(v)] : "#eee";
      ctx.fillRect(i * cw, (ny - 1 - j) * ch, cw + 1, ch + 1);
    }
  }

  // printed numbers, thinned to a readable sample — same idea as DIALux's
  // own value sheets, which never print every single point either.
  ctx.font = Math.max(8, Math.min(11, cw * 0.42)) + "px ui-monospace,Consolas,monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let j = Math.floor(labelStepY / 2); j < ny; j += labelStepY) {
    for (let i = Math.floor(labelStepX / 2); i < nx; i += labelStepX) {
      const v = values[j][i];
      if (!isFinite(v)) continue;
      const band = bandOf(v);
      const px = (i + 0.5) * cw, py = (ny - 1 - j + 0.5) * ch;
      const t = band / Math.max(1, bandColours.length - 1);
      ctx.fillStyle = t < 0.45 ? "#ffffff" : "#111111";
      ctx.fillText(Math.round(v), px, py);
    }
  }

  // mark the darkest / brightest point, the way DIALux circles them
  const [x0, y0, x1, y1] = extent;
  function marker(pt, colour, label) {
    if (!pt) return;
    const px = (pt.x - x0) / (x1 - x0) * cssW;
    const py = cssH - (pt.y - y0) / (y1 - y0) * cssH;
    ctx.beginPath(); ctx.arc(px, py, 7, 0, 2 * Math.PI);
    ctx.lineWidth = 2.2; ctx.strokeStyle = colour; ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 2, 0, 2 * Math.PI);
    ctx.fillStyle = colour; ctx.fill();
    ctx.font = "bold 10px -apple-system,Segoe UI,sans-serif";
    ctx.fillStyle = colour; ctx.textAlign = "left";
    ctx.fillText(label, px + 9, py - 8);
  }
  if (extrema) {
    marker(extrema.min, "#1b1464", "MIN " + Math.round(extrema.min.value) + " lx");
    marker(extrema.max, "#d63333", "MAX " + Math.round(extrema.max.value) + " lx");
  }
}
"""


def _thin_steps(nx: int, ny: int, max_cols: int = 16, max_rows: int = 12):
    return max(1, int(np.ceil(nx / max_cols))), max(1, int(np.ceil(ny / max_rows)))


def _legend_html(levels: List[float], colours: List[str]) -> str:
    strip = "".join(f'<div style="background:{c}"></div>' for c in colours)
    labels = "".join(f"<span>{v:.0f}</span>" for v in levels)
    return (f'<div class="legend-strip">{strip}</div>'
            f'<div class="legend-labels">{labels}</div>'
            f'<div class="hint">Illuminance bands (lx), same false-colour '
            f'scale used in the PDF/PNG report.</div>')


def _metrics_table(m: dict) -> str:
    def row(label, key, fmt="{:.1f}", suffix=""):
        v = m.get(key)
        return (f"<tr><td>{label}</td><td>{fmt.format(v)}{suffix}</td></tr>"
                if v is not None else f"<tr><td>{label}</td><td>—</td></tr>")

    return f"""<table class="metrics">
      {row("Average illuminance (Eav)", "Eav", suffix=" lx")}
      {row("Minimum (Emin)", "Emin", suffix=" lx")}
      {row("Maximum (Emax)", "Emax", suffix=" lx")}
      {row("Uniformity g1 (Emin / Eav)", "u0", fmt="{:.2f}")}
      {row("Uniformity g2 (Emin / Emax)", "Emin/Emax", fmt="{:.2f}")}
      {row("Emax / Eav", "Emax/Eav", fmt="{:.2f}")}
      {row("Median", "median", suffix=" lx")}
      {row("Std. deviation", "std", suffix=" lx")}
      {row("Calculation points", "points", fmt="{:.0f}")}
    </table>"""


def _point_box(label: str, colour: str, pt: dict) -> str:
    if not pt:
        return f'<div class="pointbox"><b>{label}</b>not available</div>'
    return (f'<div class="pointbox"><b><span class="dot" '
            f'style="background:{colour}"></span>{label}</b>'
            f'{pt["value"]:.1f} lx at x = {pt["x"]:.2f} m, y = {pt["y"]:.2f} m</div>')


def _grid_section(idx: int, grid: CalcGrid) -> str:
    disp = _downsample(grid.values)
    ny, nx = disp.shape
    vmin, vmax = float(np.nanmin(disp)), float(np.nanmax(disp))
    levels = nice_levels(vmin, vmax, n=12)
    colours = band_hex_colours(levels)
    step_x, step_y = _thin_steps(nx, ny)
    extent = list(grid.extent or (0, 0, grid.nx, grid.ny))
    metrics = grid.metrics()
    extrema = grid.extrema()

    payload = {
        "values": np.round(disp, 1).tolist(), "nx": nx, "ny": ny,
        "extent": extent, "levels": levels, "colours": colours,
        "stepX": step_x, "stepY": step_y, "extrema": extrema,
    }
    canvas_id = f"calcCanvas{idx}"

    return f"""
<section>
  <h2>{grid.name or f"Surface {idx + 1}"} — calculation results</h2>
  <div class="calc">
    <div class="stage">
      <canvas id="{canvas_id}"></canvas>
      {_legend_html(levels, colours)}
      <p class="hint">Values are the recovered illuminance in lux at each
      sampled point, printed on the exact band colour it falls into — the
      same discrete DIALux-style false-colour scale used throughout
      evostudy, not a smooth gradient.</p>
    </div>
    <div class="side">
      {_metrics_table(metrics)}
      {_point_box("Darkest point (Emin)", "#1b1464", extrema.get("min"))}
      {_point_box("Brightest point (Emax)", "#d63333", extrema.get("max"))}
    </div>
  </div>
  <script>
    (function(){{
      const d = {json.dumps(payload)};
      drawValueChart(document.getElementById("{canvas_id}"), d.values, d.nx, d.ny,
                     d.extent, d.levels, d.colours, d.stepX, d.stepY, d.extrema);
      window.addEventListener("resize", function(){{
        drawValueChart(document.getElementById("{canvas_id}"), d.values, d.nx, d.ny,
                       d.extent, d.levels, d.colours, d.stepX, d.stepY, d.extrema);
      }});
    }})();
  </script>
</section>
"""


def build_calculations_page(study: Study) -> str:
    """Render the standalone calculations.html for `study`."""
    sections = "".join(
        _grid_section(i, g) for i, g in enumerate(study.grids) if g.values is not None
    )
    if not sections:
        sections = ("<section><h2>No calculation grids</h2>"
                    "<p class='hint'>No illuminance grid was recovered for "
                    "this project — see the dashboard's notes section for "
                    "why.</p></section>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{study.project_name or "Lighting study"} — calculations</title>
<style>{CSS}</style>
<script>{JS}</script>
</head>
<body>
<header>
  <h1>{study.project_name or "Lighting study"}</h1>
  <div class="sub">Calculation results — printed values and point statistics</div>
</header>
{page_nav_html("calculations.html")}
<main>
  {sections}
</main>
<footer>
  Generated by evostudy from {study.source_path or "the archive"}. Values are
  statistically recovered from DIALux's undocumented .rsl files — cross-check
  headline numbers against DIALux itself.
</footer>
</body>
</html>"""
