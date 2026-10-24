"""Heartbeat / liveness for batch runs.

A running batch periodically rewrites ``heartbeat.json`` in its batch
directory: {batch_id, run_id, stage, progress, total, ts}. An external
watchdog (the pattern proven 2026-10-14) polls the file's mtime and
contents to tell "still working" from "died mid-stage" — process
death leaves batch.json/state.json untouched, so the heartbeat is
the only artifact that goes stale exactly when the runner dies.

Writing is atomic (tmp file + os.replace) so a poller never reads a
half-written file, and best-effort: a heartbeat failure must never
fail the batch, so all errors are swallowed here.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def beat(batch_dir: Path, batch_id: str, run_id: str, stage: str,
         progress: int | None = None, total: int | None = None) -> None:
    """Rewrite heartbeat.json for a batch. Never raises."""
    try:
        batch_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_id": batch_id,
            "run_id": run_id,
            "stage": stage,
            "progress": progress,
            "total": total,
            "ts": datetime.now(UTC).isoformat(),
        }
        tmp = batch_dir / ".heartbeat.json.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, batch_dir / "heartbeat.json")
    except Exception:
        pass
