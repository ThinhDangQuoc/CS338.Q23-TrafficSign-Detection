"""Knowledge-base and SQLite persistence helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_sign_app.config import CLASSES_PATH, DB_PATH, SIGNS_PATH


def load_classes(path: str | Path = CLASSES_PATH) -> list[str]:
    """Load class names in class_id order."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def load_sign_data(path: str | Path = SIGNS_PATH) -> dict[str, dict[str, Any]]:
    """Load traffic sign knowledge by class_id string."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_sign_info(class_id, signs_data: dict | None = None) -> dict[str, Any]:
    """Return sign info with a friendly fallback for unknown class ids."""
    try:
        normalized_id = int(class_id)
    except (TypeError, ValueError):
        normalized_id = -1

    data = signs_data if signs_data is not None else load_sign_data()
    info = (data or {}).get(str(normalized_id))
    if info:
        return info

    return {
        "class_id": normalized_id,
        "code": f"Class {normalized_id}",
        "class_name": f"Class {normalized_id}",
        "short_name": f"Biển báo class {normalized_id}",
        "type": "Chưa có dữ liệu",
        "meaning": "Chưa có thông tin chi tiết cho class_id này trong data/signs.json.",
        "driver_action": "Hãy kiểm tra lại dữ liệu biển báo và quan sát tình huống thực tế.",
        "warning": "Phát hiện biển báo chưa có dữ liệu giải thích.",
        "common_violation": "Chưa có dữ liệu lỗi thường gặp.",
        "penalty": "Chưa có dữ liệu mức phạt tham khảo.",
        "example": "Cần bổ sung nội dung cho class_id này trong signs.json.",
    }


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create the SQLite database and detection table if needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER,
                sign_name TEXT,
                confidence REAL,
                meaning TEXT,
                source TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_detection(
    class_id,
    sign_name: str,
    confidence: float,
    meaning: str,
    source: str,
    db_path: str | Path = DB_PATH,
) -> int:
    """Save one detection row and return its id."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO detections (class_id, sign_name, confidence, meaning, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(class_id), sign_name, float(confidence), meaning, source),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_detection_history(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return detection history as a DataFrame."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM detections ORDER BY detected_at DESC, id DESC",
            conn,
        )


def clear_history(db_path: str | Path = DB_PATH) -> None:
    """Remove all detection history."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM detections")
        conn.commit()

