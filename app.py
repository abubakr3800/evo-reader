"""
evostudy GUI — a plain browser front-end around the evostudy CLI toolkit,
with a live progress bar (real stages + real percentage, not a fake ramp).

Run:
    pip install flask numpy matplotlib
    python app.py
    open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

# If a job's status hasn't been updated in this long and it isn't marked
# done, assume the worker process that was running it got recycled by
# Passenger (restarts on file changes, idle timeouts, etc. are normal on
# shared hosting) and its daemon thread died with it, rather than leaving
# the person staring at a progress bar that will never move again.
STALL_TIMEOUT_S = 90

import matplotlib
matplotlib.use("Agg")  # headless — the exports run in a background thread,
                        # and an interactive backend can hang forever there
                        # instead of erroring. Set this before anything else
                        # gets a chance to import pyplot with a different backend.

from flask import (Flask, jsonify, redirect, render_template, request,
                    send_from_directory, url_for)

from evostudy.archive import EvoArchive
from evostudy.pipeline import (export_csv, export_dashboard, export_json,
                                export_per_fixture, export_report, load_study)

import os

BASE = Path(__file__).parent

# Extraction writes hundreds of files per job, fast, inside a single
# directory tree. On cPanel-style Passenger hosting, if that tree sits
# inside the app directory itself, Passenger's file-change watcher can
# see it as "the app changed" and recycle the worker mid-job — killing
# the daemon thread doing the extraction with no exception ever raised
# (see set_status()/JOBS comments above, and _interrupted_response()
# below, for what that looks like from the outside).
#
# Fix: keep upload/extraction data in a directory that is NOT inside
# BASE, so nothing Passenger watches for this app changes underneath
# it. Override with the EVOSTUDY_UPLOAD_ROOT env var if your host's
# layout needs something else (e.g. cPanel's "Setup Python App" lets
# you set custom env vars per app). Default: a sibling directory next
# to the app directory, e.g. if the app lives in
# ~/myapp/ this becomes ~/myapp_uploads/ — one level up and out of the
# tree Passenger is watching.
UPLOAD_ROOT = Path(
    os.environ.get("EVOSTUDY_UPLOAD_ROOT")
    or (BASE.parent / f"{BASE.name}_uploads")
)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# In-memory job status: job_id -> {stage, percent, done, error, orig_name}
#
# IMPORTANT if you're running this behind Passenger/mod_wsgi (cPanel-style
# shared hosting) rather than `python app.py` locally: this dict lives in
# ONE process's memory. Passenger commonly runs more than one worker
# process for the same app, and load-balances requests across them, and it
# restarts a worker whenever it notices a file under the app changed. Your
# /upload request (and the background thread it starts) lands in whichever
# worker handled it; a later /status or /progress poll can easily land in a
# *different* worker that has never heard of that job_id — which is exactly
# what an "unknown job" / stuck-at-some-% / "Failed" progress bar means.
# It isn't the pipeline crashing; the poll just can't see the worker that's
# running it. Persisting status to a file per job (below) fixes the "wrong
# worker" case. It can't fix a mid-job worker *restart* killing the
# background thread outright — see the note on run_pipeline for that.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _status_file(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def set_status(job_id: str, stage: str, percent: float, **extra):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(
            stage=stage, percent=round(min(max(percent, 0), 100), 1),
            updated_at=time.time(), **extra)
        snapshot = dict(JOBS[job_id])
    # Best-effort disk mirror so a *different* worker process — or this same
    # process after a restart — can still answer /status accurately instead
    # of returning "unknown job". Never let a write failure break the job.
    try:
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        _status_file(job_id).write_text(json.dumps(snapshot))
    except OSError:
        pass


def touch(job_id: str):
    """Refresh updated_at ONLY — stage/percent/done/error are left exactly
    as they were. Used by the heartbeat below so a single long, silent
    step (see start_heartbeat) doesn't get mistaken for a dead worker; it
    never fabricates progress that didn't happen."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["updated_at"] = time.time()
        snapshot = dict(job)
    try:
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        _status_file(job_id).write_text(json.dumps(snapshot))
    except OSError:
        pass


# Some pipeline steps (parsing a big ProjectData.xml, recovering luminaires
# from a STEP-format ProjectData.dat, rasterising one very large .rsl
# result group) are pure-CPU work with no natural place to call
# set_status() partway through. If one of those takes longer than
# STALL_TIMEOUT_S, /status's "hasn't been updated recently" check reports
# "job status lost or stalled (process restart)" even though the worker is
# still alive and working — a false positive, not the Passenger-recycle
# case this check exists for. A heartbeat thread that pings touch() every
# few seconds closes that gap: if the process is genuinely killed, the
# heartbeat dies with it and the real stall is still caught; if the
# process is just busy, updated_at keeps refreshing and the UI keeps
# waiting instead of giving up early.
def start_heartbeat(job_id: str, interval: float = 10.0) -> threading.Event:
    stop = threading.Event()

    def _beat():
        while not stop.wait(interval):
            touch(job_id)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def get_status(job_id: str) -> dict | None:
    """Memory first (freshest, and works even before the file is flushed),
    falling back to the on-disk mirror for a job this process never saw
    start — the case that produces "unknown job" with memory alone."""
    with JOBS_LOCK:
        live = JOBS.get(job_id)
    if live is not None:
        return live
    sf = _status_file(job_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except (OSError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------- helpers

def job_dir(job_id: str) -> Path:
    return UPLOAD_ROOT / job_id


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def build_file_table(archive: EvoArchive) -> list[dict]:
    """Same listing evostudy's `inspect` command prints, as row dicts for the UI."""
    rows = []
    for e in archive:
        ext, desc = e.sniff()
        rows.append({
            "path": e.path, "size": human_size(e.size),
            "size_bytes": e.size, "type": ext, "description": desc,
        })
    rows.sort(key=lambda r: r["path"])
    return rows


def list_jobs() -> list[dict]:
    """Every job that has a folder under uploads/, newest first — read off
    disk (not just the in-memory JOBS dict) so past runs still show up
    after a restart, even though their live progress info is gone."""
    jobs = []
    if not UPLOAD_ROOT.exists():
        return jobs

    for jdir in UPLOAD_ROOT.iterdir():
        if not jdir.is_dir():
            continue
        job_id = jdir.name

        manifest = jdir / "manifest_note.txt"
        orig_name = manifest.read_text().strip() if manifest.exists() else "(unknown file)"

        live = get_status(job_id)

        out_dir = jdir / "output"
        has_output = out_dir.exists() and any(out_dir.iterdir())
        has_error_file = (jdir / "pipeline_error.txt").exists()

        if live is not None and not live.get("done"):
            status, stage = "running", live.get("stage", "Running…")
        elif (live is not None and live.get("error")) or has_error_file:
            status, stage = "failed", "Failed"
        elif has_output:
            status, stage = "done", "Done"
        else:
            # No live entry (server restarted) and nothing in output/ yet —
            # most likely an upload that never finished.
            status, stage = "unknown", "No output found — may not have finished"

        try:
            mtime = jdir.stat().st_mtime
        except OSError:
            mtime = 0

        jobs.append({
            "job_id": job_id,
            "orig_name": orig_name,
            "status": status,
            "stage": stage,
            "when": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "",
            "mtime": mtime,
        })

    jobs.sort(key=lambda j: j["mtime"], reverse=True)
    return jobs


# --------------------------------------------------------------------------- the actual work, run in a background thread

def run_pipeline(job_id: str, src_path: Path, orig_name: str,
                 skip_3d: bool = False, make_pdf: bool = False,
                 make_images: bool = False):
    """
    By default this only produces the dashboard + JSON/CSV data — all real
    recovered numbers, nothing rendered as an image. The PDF report and the
    PNG image sheets (including per-fixture image exports) are each opt-in
    independently (`make_pdf=True`, `make_images=True`), and can also be
    requested later — separately or together — via /generate_report/<job_id>
    once you've actually looked at the numbers and decided you want them.
    """
    jdir = job_dir(job_id)
    heartbeat_stop = start_heartbeat(job_id)
    try:
        set_status(job_id, f"Converting {orig_name} to a .zip twin", 2)
        zip_twin = jdir / (Path(orig_name).stem + ".evo.zip")
        shutil.copyfile(src_path, zip_twin)

        set_status(job_id, "Extracting the archive", 8)
        extract_dir = jdir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(str(zip_twin), str(extract_dir), format="zip")
        set_status(job_id, "Extracted", 15)

        set_status(job_id, "Reading and identifying every file", 17)
        archive = EvoArchive.open(str(src_path))
        n_files = len(archive)
        set_status(job_id, f"Identified {n_files} files", 20)

        def on_progress(msg, frac):
            set_status(job_id, msg, 20 + 50 * frac)

        set_status(job_id, "Parsing project data", 20)
        study = load_study(str(src_path), progress=on_progress)

        out_dir = jdir / "output"
        out_dir.mkdir(exist_ok=True)

        set_status(job_id, "Building the interactive dashboard", 80)
        export_dashboard(study, str(out_dir / "dashboard.html"))

        set_status(job_id, "Writing JSON + CSV data", 92)
        export_json(study, str(out_dir / "study.json"))
        export_csv(study, str(out_dir))

        if make_pdf or make_images:
            _render_report(job_id, study, out_dir, skip_3d, make_pdf, make_images)

        set_status(job_id, "Done", 100, done=True)

    except Exception:
        tb = traceback.format_exc()
        (jdir / "pipeline_error.txt").write_text(tb)
        set_status(job_id, "Failed", 100, done=True, error=tb)
    finally:
        heartbeat_stop.set()


def _render_report(job_id: str, study, out_dir: Path, skip_3d: bool,
                    make_pdf: bool = True, make_images: bool = True):
    """The slow, image-producing half of the pipeline — the PDF report
    and/or the PNG image sheets. `make_pdf` and `make_images` are
    independent switches, so a run can produce just the PDF, just the
    images, or both. Only called when explicitly requested, either at
    upload time or later via /generate_report."""
    def on_sheet_progress(done, total, name):
        set_status(job_id, f"Rendering report ({done}/{total}): {name}",
                  75 + 20 * (done / total if total else 1.0))

    label = " + ".join(p for p in (("PDF" if make_pdf else None),
                                    ("PNG" if make_images else None)) if p)
    set_status(job_id, f"Rendering the report ({label})", 75)
    export_report(study,
                  pdf_path=(out_dir / "study.pdf") if make_pdf else None,
                  png_dir=(out_dir / "sheets") if make_images else None,
                  include_3d=(not skip_3d) if (make_pdf or make_images) else False,
                  progress=on_sheet_progress)

    if make_images:
        set_status(job_id, "Exporting per-fixture views", 98)
        try:
            export_per_fixture(study, str(out_dir / "per_fixture_data"))
        except Exception:
            pass  # not every study has enough recovered geometry for this


def _reload_study(job_id: str):
    """Re-run the (fast) parse step to get a Study object back for a job
    whose original in-memory result no longer exists — used by the
    on-demand report generator, which runs in its own request rather than
    keeping every Study alive in memory for the life of the process."""
    jdir = job_dir(job_id)
    orig_name = (jdir / "manifest_note.txt").read_text().strip()
    return load_study(str(jdir / orig_name))


# --------------------------------------------------------------------------- routes

@app.route("/")
def index():
    return render_template("index.html", jobs=list_jobs())


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("evo_file")
    if not f or not f.filename:
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:12]
    jdir = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    orig_name = f.filename
    src_path = jdir / orig_name
    f.save(src_path)
    (jdir / "manifest_note.txt").write_text(orig_name)
    skip_3d = request.form.get("skip_3d") == "1"
    make_pdf = request.form.get("make_pdf") == "1"
    make_images = request.form.get("make_images") == "1"

    set_status(job_id, "Queued", 0, done=False, error=None, orig_name=orig_name)
    threading.Thread(target=run_pipeline,
                     args=(job_id, src_path, orig_name, skip_3d, make_pdf, make_images),
                     daemon=True).start()

    return redirect(url_for("progress_page", job_id=job_id))


@app.route("/generate_report/<job_id>", methods=["POST"])
def generate_report(job_id):
    """On-demand PDF and/or PNG report generation for a job that was run
    without one — nothing renders as an image until you ask, and you can
    ask for just the PDF, just the PNG sheets, or both."""
    jdir = job_dir(job_id)
    if not jdir.exists():
        return "Unknown job.", 404
    try:
        study = _reload_study(job_id)
        out_dir = jdir / "output"
        out_dir.mkdir(exist_ok=True)
        skip_3d = request.form.get("skip_3d") == "1"
        make_pdf = request.form.get("make_pdf") == "1"
        make_images = request.form.get("make_images") == "1"
        set_status(job_id, "Rendering report", 75, done=False)
        _render_report(job_id, study, out_dir, skip_3d, make_pdf, make_images)
        set_status(job_id, "Done", 100, done=True)
    except Exception:
        tb = traceback.format_exc()
        (jdir / "pipeline_error.txt").write_text(tb)
        set_status(job_id, "Failed", 100, done=True, error=tb)
    return redirect(url_for("result", job_id=job_id))


@app.route("/progress/<job_id>")
def progress_page(job_id):
    if not job_dir(job_id).exists():
        return "Unknown job.", 404
    return render_template("progress.html", job_id=job_id)


def _interrupted_response():
    return jsonify({"stage": "Interrupted — the app process was likely "
                     "restarted or recycled mid-job (common on shared "
                     "hosting). Please re-upload.",
                     "percent": 0, "done": True,
                     "error": "job status lost or stalled (process restart)"})


@app.route("/status/<job_id>")
def status(job_id):
    s = get_status(job_id)
    if s is None:
        if job_dir(job_id).exists():
            # The folder exists but no status was ever recorded for it (or
            # the file got removed) — most likely the worker that started
            # this job was restarted mid-run and its daemon thread died
            # with it. Say that plainly instead of a bare 404, since "the
            # pipeline crashed" and "the process got recycled" need
            # different fixes.
            return _interrupted_response()
        return jsonify({"error": "unknown job"}), 404
    if not s.get("done") and time.time() - s.get("updated_at", 0) > STALL_TIMEOUT_S:
        return _interrupted_response()
    return jsonify(s)


@app.route("/result/<job_id>")
def result(job_id):
    jdir = job_dir(job_id)
    if not jdir.exists():
        return "Unknown job.", 404

    pipeline_error = None
    if (jdir / "pipeline_error.txt").exists():
        pipeline_error = (jdir / "pipeline_error.txt").read_text()

    orig_name = (jdir / "manifest_note.txt").read_text().strip() if (jdir / "manifest_note.txt").exists() else "project.evo"

    file_rows = []
    src_path = jdir / orig_name
    if src_path.exists():
        try:
            archive = EvoArchive.open(str(src_path))
            file_rows = build_file_table(archive)
        except Exception:
            pass

    out_dir = jdir / "output"
    output_rows = []
    has_dashboard = (out_dir / "dashboard.html").exists()
    has_pdf = (out_dir / "study.pdf").exists()
    sheets_dir = out_dir / "sheets"
    has_images = sheets_dir.exists() and any(sheets_dir.iterdir())
    study_json = {}
    if out_dir.exists():
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                sz = p.stat().st_size
                output_rows.append({
                    "path": str(p.relative_to(out_dir)), "size": human_size(sz),
                    "size_bytes": sz,
                })
        import json
        sj = out_dir / "study.json"
        if sj.exists():
            try:
                study_json = json.loads(sj.read_text())
            except Exception:
                study_json = {}

    return render_template(
        "result.html", job_id=job_id, orig_name=orig_name, error=None,
        pipeline_error=pipeline_error, file_rows=file_rows, output_rows=output_rows,
        has_dashboard=has_dashboard, has_pdf=has_pdf, has_images=has_images,
        study_json=study_json,
    )


@app.route("/files/<job_id>/<path:filename>")
def serve_output(job_id, filename):
    return send_from_directory(job_dir(job_id) / "output", filename)


@app.route("/dashboard/<job_id>")
def dashboard(job_id):
    return send_from_directory(job_dir(job_id) / "output", "dashboard.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)