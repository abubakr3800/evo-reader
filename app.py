"""
evostudy GUI — a plain browser front-end around the evostudy CLI toolkit,
with a live progress bar (real stages + real percentage, not a fake ramp).

Run:
    pip install flask numpy matplotlib
    python app.py
    open http://127.0.0.1:5000

IMPORTANT (Passenger/cPanel deployment): this process only ENQUEUES jobs —
it never runs the pipeline itself. A separate process, worker.py, must be
running (via cron — see worker.py's docstring) to actually process them.
See jobs.py for why: a background thread living inside a Passenger web
worker dies silently whenever Passenger recycles that worker, which it can
do for reasons unrelated to your job (idle-pool reclaim, per-account
memory/CPU limits, a restart.txt touch). worker.py runs outside that pool
entirely, so it isn't subject to those reclaims.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from flask import (Flask, jsonify, redirect, render_template, request,
                    send_from_directory, url_for)

import jobs
from evostudy.archive import EvoArchive
from jobs import (STALL_TIMEOUT_S, build_file_table, get_status,
                   human_size, job_dir, list_jobs)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB


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

    # Hand off to the queue — worker.py (running under cron, outside
    # Passenger's pool) is what actually processes it. See jobs.py.
    jobs.enqueue(job_id, src_path, orig_name, skip_3d, make_pdf, make_images)

    return redirect(url_for("progress_page", job_id=job_id))


@app.route("/generate_report/<job_id>", methods=["POST"])
def generate_report(job_id):
    """On-demand PDF and/or PNG report generation for a job that was run
    without one — nothing renders as an image until you ask, and you can
    ask for just the PDF, just the PNG sheets, or both.

    Queued (not run inline) for the same reason as the main pipeline —
    see jobs.py. Sends you to the progress page since this can take a
    while on a large study; it'll bounce you to /result when done."""
    jdir = job_dir(job_id)
    if not jdir.exists():
        return "Unknown job.", 404
    skip_3d = request.form.get("skip_3d") == "1"
    make_pdf = request.form.get("make_pdf") == "1"
    make_images = request.form.get("make_images") == "1"
    jobs.enqueue_report(job_id, skip_3d, make_pdf, make_images)
    return redirect(url_for("progress_page", job_id=job_id))


@app.route("/delete/<job_id>", methods=["POST"])
def delete_job(job_id):
    """Stops a running job (kills the worker process handling it — cron
    starts a fresh one within 5 min for any other queued work) or just
    removes a queued/finished one, then deletes its files."""
    if not job_dir(job_id).exists() and not (jobs.QUEUE_DIR / f"{job_id}.json").exists():
        return "Unknown job.", 404
    jobs.cancel_job(job_id)
    jobs.delete_job_files(job_id)
    return redirect(url_for("index"))


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