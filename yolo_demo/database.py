"""SQLite helpers for detection history."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "detections.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sign_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                warning_message TEXT NOT NULL,
                snapshot_path TEXT,
                source_type TEXT NOT NULL,
                detected_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_detection(
    sign_name: str,
    confidence: float,
    warning_message: str,
    snapshot_path: str | None,
    source_type: str,
) -> int:
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO detections
                (sign_name, confidence, warning_message, snapshot_path, source_type, detected_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sign_name, confidence, warning_message, snapshot_path, source_type, detected_at),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_detection_rows(limit: int | None = 100) -> list[dict[str, Any]]:
    query = "SELECT * FROM detections ORDER BY id DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_stats() -> dict[str, Any]:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM detections").fetchone()["count"]
        by_class_rows = conn.execute(
            """
            SELECT sign_name, COUNT(*) AS count
            FROM detections
            GROUP BY sign_name
            ORDER BY count DESC, sign_name ASC
            """
        ).fetchall()

    by_class = [{"sign_name": row["sign_name"], "count": row["count"]} for row in by_class_rows]
    top_sign = by_class[0] if by_class else None
    return {
        "total": total,
        "by_class": by_class,
        "top_sign": top_sign,
    }
