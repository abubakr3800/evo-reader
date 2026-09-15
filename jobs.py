"""
Job engine: status tracking + the actual pipeline work.

Deliberately has NO Flask import. app.py (the web front end) and worker.py
(the cron-driven background processor) both import this module. Keeping it
Flask-free means worker.py can run as a plain `python worker.py` process
under cron/SSH, completely outside anything Passenger manages — which is
the whole point: Passenger can recycle/kill the web app's worker processes
for reasons that have nothing to do with your job (idle pool reclaim,
per-account memory/CPU limits, a restart.txt touch), and when it does, any
background thread living inside that process dies with it, silently, with
no exception for Python to catch. A cron-launched process is not part of
that pool and is not subject to those reclaim rules.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

STALL_TIMEOUT_S = 90

import matplotlib
matplotlib.use("Agg")

from evostudy.archive import EvoArchive
from evostudy.pipeline import (export_csv, export_dashboard, export_json,
                                export_per_fixture, export_report, load_study)

BASE = Path(__file__).parent

# Upload/extraction/output data: kept OUTSIDE BASE so Passenger's file-change
# watcher (which triggers a restart when files under the app root change)
# never sees it.
UPLOAD_ROOT = Path(
    os.environ.get("EVOSTUDY_UPLOAD_ROOT")
    or (BASE.parent / f"{BASE.name}_uploads")
)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Queue: one JSON file per job waiting to be picked up by worker.py, also
# OUTSIDE BASE for the same reason. /upload just drops a file here;
# worker.py claims it with an atomic rename before working on it, so two
# worker instances (e.g. a brief overlap during cron's own restart check)
# can never double-process the same job.
QUEUE_DIR = Path(
    os.environ.get("EVOSTUDY_QUEUE_ROOT")
    or (BASE.parent / f"{BASE.name}_queue")
)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def job_dir(job_id: str) -> Path:
    return UPLOAD_ROOT / job_id


def _status_file(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def set_status(job_id: str, stage: str, percent: float, **extra):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(
            stage=stage, percent=round(min(max(percent, 0), 100), 1),
            updated_at=time.time(), **extra)
        snapshot = dict(JOBS[job_id])
    try:
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        _status_file(job_id).write_text(json.dumps(snapshot))
    except OSError:
        pass


def touch(job_id: str):
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


def start_heartbeat(job_id: str, interval: float = 10.0) -> threading.Event:
    stop = threading.Event()

    def _beat():
        while not stop.wait(interval):
            touch(job_id)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def get_status(job_id: str) -> dict | None:
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


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def build_file_table(archive: EvoArchive) -> list[dict]:
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
    jobs = []
    if not UPLOAD_ROOT.exists():
        return jobs

    queued_ids = {p.stem for p in QUEUE_DIR.glob("*.json")}

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

        if job_id in queued_ids and live is None:
            status, stage = "queued", "Waiting for worker…"
        elif live is not None and not live.get("done"):
            status, stage = "running", live.get("stage", "Running…")
        elif (live is not None and live.get("error")) or has_error_file:
            status, stage = "failed", "Failed"
        elif has_output:
            status, stage = "done", "Done"
        else:
            status, stage = "unknown", "No output found — may not have finished"

        try:
            mtime = jdir.stat().st_mtime
        except OSError:
            mtime = 0

        jobs.append({
            "job_id": job_id, "orig_name": orig_name, "status": status,
            "stage": stage,
            "when": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "",
            "mtime": mtime,
        })

    jobs.sort(key=lambda j: j["mtime"], reverse=True)
    return jobs


def enqueue(job_id: str, src_path: Path, orig_name: str,
            skip_3d: bool = False, make_pdf: bool = False,
            make_images: bool = False):
    """Called by the web app's /upload route. Does NOT run anything —
    just drops a request file in QUEUE_DIR for worker.py to pick up."""
    set_status(job_id, "Queued — waiting for worker", 0,
               done=False, error=None, orig_name=orig_name)
    req = {
        "kind": "pipeline",
        "job_id": job_id, "src_path": str(src_path), "orig_name": orig_name,
        "skip_3d": skip_3d, "make_pdf": make_pdf, "make_images": make_images,
    }
    _write_queue_entry(job_id, req)


def enqueue_report(job_id: str, skip_3d: bool = False,
                    make_pdf: bool = False, make_images: bool = False):
    """Called by /generate_report. Same queue as enqueue() — a report
    render is just as CPU/memory-heavy (matplotlib figures) as the main
    pipeline and was previously run synchronously inside the request,
    which is worse: it ties up a Passenger worker for however long
    rendering takes and risks a proxy/Apache timeout killing the
    connection on top of the Passenger-recycle risk."""
    set_status(job_id, "Queued — waiting for worker (report)", 75,
               done=False, error=None)
    req = {
        "kind": "report",
        "job_id": job_id,
        "skip_3d": skip_3d, "make_pdf": make_pdf, "make_images": make_images,
    }
    _write_queue_entry(job_id, req)


def _write_queue_entry(job_id: str, req: dict):
    tmp = QUEUE_DIR / f"{job_id}.json.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(QUEUE_DIR / f"{job_id}.json")  # atomic — worker never sees a half-written file


def _current_file() -> Path:
    return QUEUE_DIR / "current.json"


def set_current(job_id: str | None):
    """Records which job_id the worker process is actively running right
    now, plus its own PID, so /delete can find and interrupt it. Called by
    worker.py; job_id=None clears it once a job finishes."""
    cf = _current_file()
    if job_id is None:
        cf.unlink(missing_ok=True)
        return
    cf.write_text(json.dumps({"job_id": job_id, "pid": os.getpid()}))


def get_current() -> dict | None:
    cf = _current_file()
    if not cf.exists():
        return None
    try:
        return json.loads(cf.read_text())
    except (OSError, ValueError):
        return None


def cancel_job(job_id: str) -> str:
    """Called by the web app's /delete route. Returns what it did:
    'dequeued' (was only queued, never started — just removed),
    'killed' (was actively running — the worker process handling it was
    terminated; cron will start a fresh worker within 5 min for any other
    queued jobs), or 'none' (nothing to stop, just cleanup)."""
    q = QUEUE_DIR / f"{job_id}.json"
    if q.exists():
        q.unlink(missing_ok=True)
        return "dequeued"

    current = get_current()
    if current and current.get("job_id") == job_id:
        pid = current.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                os.kill(pid, signal.SIGKILL)  # in case SIGTERM didn't land fast enough
            except ProcessLookupError:
                pass  # already gone
        _current_file().unlink(missing_ok=True)
        return "killed"

    return "none"


def delete_job_files(job_id: str):
    jdir = job_dir(job_id)
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)


def claim_next_job() -> dict | None:
    """Called only by worker.py. Atomically claims the oldest queued job
    (rename is atomic on POSIX, so even a brief overlap between an old and
    a freshly cron-spawned worker can't double-claim one). Returns the
    request dict, or None if the queue is empty."""
    pending = sorted(QUEUE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in pending:
        claimed = p.with_suffix(".json.claimed")
        try:
            p.rename(claimed)
        except OSError:
            continue  # another worker got it first
        try:
            req = json.loads(claimed.read_text())
        finally:
            claimed.unlink(missing_ok=True)
        return req
    return None


# --------------------------------------------------------------------------- the actual work

def run_pipeline(job_id: str, src_path: Path, orig_name: str,
                 skip_3d: bool = False, make_pdf: bool = False,
                 make_images: bool = False):
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
            pass


def reload_study(job_id: str):
    jdir = job_dir(job_id)
    orig_name = (jdir / "manifest_note.txt").read_text().strip()
    return load_study(str(jdir / orig_name))


def run_report(job_id: str, skip_3d: bool = False,
                make_pdf: bool = False, make_images: bool = False):
    """The queued equivalent of the old synchronous /generate_report body.
    Same heartbeat + error handling as run_pipeline, for the same reason:
    this is heavy enough to be worth protecting."""
    jdir = job_dir(job_id)
    heartbeat_stop = start_heartbeat(job_id)
    try:
        study = reload_study(job_id)
        out_dir = jdir / "output"
        out_dir.mkdir(exist_ok=True)
        set_status(job_id, "Rendering report", 75, done=False)
        _render_report(job_id, study, out_dir, skip_3d, make_pdf, make_images)
        set_status(job_id, "Done", 100, done=True)
    except Exception:
        tb = traceback.format_exc()
        (jdir / "pipeline_error.txt").write_text(tb)
        set_status(job_id, "Failed", 100, done=True, error=tb)
    finally:
        heartbeat_stop.set()
