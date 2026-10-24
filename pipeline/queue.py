"""SQLite-backed work queue for batch pipeline stages.

Design (§10.2): queue-based execution survives VM replacements.
- Single file (queue.db) in the batch directory — no server needed.
- Workers claim items atomically; crash → stale claims get requeued.
- Items are idempotent (keyed on batch_id + run_id + item_id).
- Failed items after max_attempts go to DLQ for inspection.

This is the prototype implementation. UC-2+ uses Prefect work queues
with the same interface.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueueItem:
    item_id: str
    stage: str
    payload: dict
    attempts: int
    status: str  # pending | claimed | done | dlq


SCHEMA = """
CREATE TABLE IF NOT EXISTS queue_items (
    item_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claimed_at REAL,
    error TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, run_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_queue_stage_status
    ON queue_items (batch_id, stage, status);
"""

# Pre-G-08-fix schema keyed rows on item_id alone, so a FAILED batch
# retried under a NEW run_id collided with the old run's rows and its
# enqueue was silently ignored (INSERT OR IGNORE) — the new run's
# stage had nothing to claim. Existing queue.db files are migrated
# on open: same columns, composite key; in-flight rows keep their
# item_id/batch_id/run_id, so a same-run resume is unaffected.
_MIGRATE_OLD_SCHEMA = """
ALTER TABLE queue_items RENAME TO queue_items_legacy;
CREATE TABLE queue_items (
    item_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claimed_at REAL,
    error TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, run_id, item_id)
);
INSERT OR IGNORE INTO queue_items SELECT * FROM queue_items_legacy;
DROP TABLE queue_items_legacy;
CREATE INDEX IF NOT EXISTS idx_queue_stage_status
    ON queue_items (batch_id, stage, status);
"""


class WorkQueue:
    """Durable work queue backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate_legacy_schema()
        self._conn.commit()

    def _migrate_legacy_schema(self) -> None:
        """Rebuild a pre-fix queue_items (item_id sole PRIMARY KEY)
        with the run-scoped composite key. No-op on new-schema DBs."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='queue_items'").fetchone()
        if row and row[0] and "item_id TEXT PRIMARY KEY" in row[0]:
            self._conn.executescript(_MIGRATE_OLD_SCHEMA)

    def close(self):
        self._conn.close()

    def enqueue(self, batch_id: str, run_id: str, stage: str,
                items: list[dict], max_attempts: int = 3) -> int:
        """Add items. Idempotent per run: re-enqueueing the same
        (batch_id, run_id, item_id) won't duplicate, while a retry
        under a NEW run_id enqueues a fresh set of rows (G-08)."""
        now = time.time()
        with self._conn:
            for it in items:
                item_id = it.get("item_id") or str(uuid.uuid4())
                self._conn.execute(
                    """INSERT OR IGNORE INTO queue_items
                       (item_id, batch_id, run_id, stage, payload,
                        status, max_attempts, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                    (item_id, batch_id, run_id, stage,
                     json.dumps(it), max_attempts, now, now))
        # Rows now queued for this (batch, run, stage)
        with self._conn:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM queue_items "
                "WHERE batch_id=? AND run_id=? AND stage=?",
                (batch_id, run_id, stage))
            return cur.fetchone()[0]

    def claim(self, batch_id: str, run_id: str, stage: str,
              worker_id: str, limit: int = 10,
              lease_seconds: int = 300) -> list[QueueItem]:
        """Atomically claim up to `limit` pending items."""
        now = time.time()
        with self._conn:
            # Requeue stale claims first (crash recovery)
            self._conn.execute(
                """UPDATE queue_items SET status='pending',
                       claimed_by=NULL, claimed_at=NULL, updated_at=?
                   WHERE batch_id=? AND stage=? AND status='claimed'
                     AND claimed_at < ?""",
                (now, batch_id, stage, now - lease_seconds))
            cur = self._conn.execute(
                """SELECT item_id, payload, attempts FROM queue_items
                   WHERE batch_id=? AND run_id=? AND stage=?
                     AND status='pending'
                   ORDER BY created_at LIMIT ?""",
                (batch_id, run_id, stage, limit))
            rows = cur.fetchall()
            items = []
            for item_id, payload, attempts in rows:
                self._conn.execute(
                    """UPDATE queue_items
                       SET status='claimed', claimed_by=?, claimed_at=?,
                           updated_at=?
                       WHERE item_id=? AND batch_id=? AND run_id=?
                         AND stage=? AND status='pending'""",
                    (worker_id, now, now, item_id,
                     batch_id, run_id, stage))
                if self._conn.total_changes:
                    items.append(QueueItem(
                        item_id=item_id, stage=stage,
                        payload=json.loads(payload),
                        attempts=attempts, status="claimed"))
            return items

    def complete(self, item_id: str, result: dict | None = None,
                 batch_id: str | None = None,
                 run_id: str | None = None) -> None:
        """Mark an item done. Pass batch_id/run_id to scope the update
        to one run's row — since the G-08 fix, another run of the same
        batch may hold a row with the same item_id."""
        where = "item_id=?"
        params: list = [json.dumps(result) if result else None,
                        time.time(), item_id]
        if batch_id is not None:
            where += " AND batch_id=?"
            params.append(batch_id)
        if run_id is not None:
            where += " AND run_id=?"
            params.append(run_id)
        with self._conn:
            self._conn.execute(
                f"""UPDATE queue_items SET status='done', result=?,
                       updated_at=? WHERE {where}""",
                params)

    def fail(self, item_id: str, error: str,
             batch_id: str | None = None,
             run_id: str | None = None) -> str:
        """Record failure. Returns new status: 'pending' (retry) or 'dlq'.
        Pass batch_id/run_id to scope to one run's row (see complete)."""
        where = "item_id=?"
        scope: list = [item_id]
        if batch_id is not None:
            where += " AND batch_id=?"
            scope.append(batch_id)
        if run_id is not None:
            where += " AND run_id=?"
            scope.append(run_id)
        with self._conn:
            cur = self._conn.execute(
                "SELECT attempts, max_attempts FROM queue_items "
                f"WHERE {where}", scope)
            row = cur.fetchone()
            if not row:
                return "unknown"
            attempts, max_attempts = row
            attempts += 1
            status = "dlq" if attempts >= max_attempts else "pending"
            self._conn.execute(
                f"""UPDATE queue_items
                   SET status=?, attempts=?, error=?,
                       claimed_by=NULL, claimed_at=NULL, updated_at=?
                   WHERE {where}""",
                [status, attempts, error[:500], time.time(), *scope])
            return status

    def depth(self, batch_id: str, run_id: str, stage: str) -> dict:
        """Remaining work per status."""
        with self._conn:
            cur = self._conn.execute(
                """SELECT status, COUNT(*) FROM queue_items
                   WHERE batch_id=? AND run_id=? AND stage=?
                   GROUP BY status""",
                (batch_id, run_id, stage))
            return {row[0]: row[1] for row in cur.fetchall()}

    def dlq_items(self, batch_id: str, stage: str = "") -> list[dict]:
        with self._conn:
            q = ("SELECT item_id, payload, attempts, error FROM queue_items "
                 "WHERE batch_id=? AND status='dlq'")
            params: list = [batch_id]
            if stage:
                q += " AND stage=?"
                params.append(stage)
            cur = self._conn.execute(q, params)
            return [{"item_id": r[0], "payload": json.loads(r[1]),
                     "attempts": r[2], "error": r[3]}
                    for r in cur.fetchall()]

    def requeue_dlq(self, batch_id: str, stage: str = "") -> int:
        """Move DLQ items back to pending (after fixing the cause)."""
        with self._conn:
            q = ("UPDATE queue_items SET status='pending', attempts=0, "
                 "error=NULL, updated_at=? "
                 "WHERE batch_id=? AND status='dlq'")
            params: list = [time.time(), batch_id]
            if stage:
                q += " AND stage=?"
                params.append(stage)
            cur = self._conn.execute(q, params)
            return cur.rowcount
