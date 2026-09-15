# evostudy GUI

A browser front-end around the `evostudy` toolkit (vendored in this folder)
for reading DIALux evo (.evo) project files without DIALux installed —
with a live progress bar tied to real pipeline stages.

## Run it

```bash
pip install flask numpy matplotlib
python app.py
```

Open http://127.0.0.1:5000, drop in a `.evo` file, and you're taken to a
progress page that polls `/status/<job_id>` every 500ms and shows the
actual stage + percentage as the background job runs:

1. Convert to `.evo.zip` + extract (0–15%)
2. Identify every file (15–20%)
3. Parse project data + recover each result group's illuminance grid
   (20–70% — this is generally the slowest part on projects with many
   result groups, and now reports per-group progress, not a flat wait)
4. Build the dashboard, PDF, JSON/CSV, PNG sheets, per-fixture exports
   (70–100%)

When it hits 100% you're redirected automatically to the results page:
summary cards, the embedded interactive dashboard, a download table for
every generated file, and the full file-by-file archive listing.

Job status lives in memory (`app.py`'s `JOBS` dict) — fine for local,
single-user use. Swap for Redis/a DB if this ever needs to survive a
restart or serve concurrent users.

## Recent fixes baked into this vendored copy

- `dashboard.html` no longer throws `initViewer is not defined` — the
  shared JS is now loaded in `<head>`, before any per-surface viewer
  script calls it.
- Each viewer layer is downsampled to ~10,000 display points before being
  embedded as JSON. A large study (many fixtures × large grid) used to
  produce a dashboard.html of tens of megabytes that was slow to open;
  now it stays under ~1MB regardless of the source grid's native
  resolution. Full-resolution values are untouched everywhere else (CSV/
  JSON export, printed Eav/Emin/Emax).

## Command-line still works

Everything here just wraps the CLI. You can still use it directly —
see `evostudy/README.md` (or run `python -m evostudy --help`).
