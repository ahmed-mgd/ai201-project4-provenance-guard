"""Persistence for Milestone 3.

Two stores, each with a clear job:
  - submissions (SQLite): the current state of each submission, including its
    mutable status. Lets the appeal endpoint (M5) look a submission up by id and
    change its status.
  - audit log (JSONL): an append-only history. One JSON object per line. This is
    what GET /log returns.

Later milestones extend both. For now we log a classification from a single
signal plus a placeholder confidence.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "provenance.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "audit.jsonl")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                content_id       TEXT PRIMARY KEY,
                text             TEXT NOT NULL,
                creator_id       TEXT,
                attribution      TEXT NOT NULL,
                probability_ai   REAL NOT NULL,
                confidence       REAL NOT NULL,
                llm_score        REAL NOT NULL,
                stylometry_score REAL NOT NULL,
                lexical_score    REAL NOT NULL,
                status           TEXT NOT NULL DEFAULT 'classified',
                created_at       TEXT NOT NULL
            );
            """
        )


def save_submission(record: dict) -> None:
    """Persist one classified submission (current state)."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO submissions
               (content_id, text, creator_id, attribution, probability_ai,
                confidence, llm_score, stylometry_score, lexical_score,
                status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["content_id"],
                record["text"],
                record.get("creator_id"),
                record["attribution"],
                record["probability_ai"],
                record["confidence"],
                record["llm_score"],
                record["stylometry_score"],
                record["lexical_score"],
                record.get("status", "classified"),
                record["created_at"],
            ),
        )


def get_submission(content_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def append_audit(entry: dict) -> None:
    """Append one JSON object as a line to the audit log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_audit_log() -> list[dict]:
    """Return all audit-log entries, parsed from JSONL."""
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
