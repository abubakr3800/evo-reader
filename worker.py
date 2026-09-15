#!/usr/bin/env python3
"""
Standalone queue worker for evostudy GUI.

Run this from cron, NOT from the web app. It lives entirely outside
Passenger's process pool, so Passenger recycling/killing your web workers
(idle reclaim, per-account memory/CPU limits, a restart.txt touch) can
never interrupt a job it's running.

Self-healing "keep exactly one instance alive" pattern, suited to shared
hosting where you don't get a real daemon/systemd service:

    * * * * * cd /home/USER/evostudy && /home/USER/venv/bin/python worker.py >> worker.log 2>&1

Every minute, cron tries to start this script. It takes a non-blocking
file lock first:
  - If it gets the lock: no other worker is running. It keeps running,
    processing jobs as they appear in the queue, for as long as the lock
    is held (i.e. indefinitely — see the main loop below).
  - If it can't get the lock: an instance from an earlier cron tick is
    still alive and working, so this one exits immediately (0.1s, no-op).

If the running instance ever dies for any reason (host reboot, OOM, you
killing it), the lock is released automatically and the very next minute's
cron tick starts a fresh one. A job that was mid-processing when the
worker died is left "claimed but incomplete" — see NOTE below.
"""

from __future__ import annotations

import fcntl
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import jobs  # noqa: E402

LOCK_PATH = jobs.QUEUE_DIR / "worker.lock"
IDLE_POLL_S = 3


def _log(msg: str):
    print(msg, flush=True)


def main():
    lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log("Another worker instance holds the lock — exiting.")
        return

    _log(f"Worker started (pid={__import__('os').getpid()}), watching {jobs.QUEUE_DIR}")
    while True:
        req = jobs.claim_next_job()
        if req is None:
            time.sleep(IDLE_POLL_S)
            continue

        kind = req.get("kind", "pipeline")
        _log(f"Processing {kind} job {req['job_id']}")
        jobs.set_current(req["job_id"])
        try:
            if kind == "report":
                jobs.run_report(
                    req["job_id"],
                    skip_3d=req.get("skip_3d", False),
                    make_pdf=req.get("make_pdf", False),
                    make_images=req.get("make_images", False),
                )
            else:
                jobs.run_pipeline(
                    req["job_id"], Path(req["src_path"]), req["orig_name"],
                    skip_3d=req.get("skip_3d", False),
                    make_pdf=req.get("make_pdf", False),
                    make_images=req.get("make_images", False),
                )
        finally:
            jobs.set_current(None)
        _log(f"Finished job {req['job_id']}")


if __name__ == "__main__":
    main()

# NOTE on a job that was mid-run when the worker process itself died:
# claim_next_job() already deleted its queue entry, so it won't be
# retried automatically. /status will correctly show it as stalled
# (STALL_TIMEOUT_S with no heartbeat) rather than hanging forever. If you
# want automatic retry instead of surfacing the stall to the user, the
# straightforward addition is: before deleting the ".json.claimed" file
# in claim_next_job(), copy it into a "<job_id>.json" in-progress marker
# under UPLOAD_ROOT/job_id/, and have worker.py's startup scan for any
# such markers left over from a previous run and re-enqueue them.
