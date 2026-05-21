"""Knowledge-base and SQLite persistence helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_sign_app.config import CLASSES_PATH, DB_PATH, PENALTIES_PATH, SIGNS_PATH


NO_PENALTY_DATA_MESSAGE = "Chưa có dữ liệu mức phạt cho biển này."
VEHICLE_NOT_APPLICABLE_MESSAGE = (
    "Biển này không áp dụng trực tiếp cho loại phương tiện đã chọn hoặc cần kiểm tra biển phụ/tình huống thực tế."
)

VEHICLE_LABELS = {
    "car": "Ô tô",
    "motorbike": "Xe máy",
    "truck": "Xe tải",
    "bus": "Xe buýt",
    "all": "Tất cả",
}

VEHICLE_ALIASES = {
    "bus": "passenger_car",
    "passenger_car": "bus",
}


def get_vehicle_label(vehicle_type: str | None) -> str:
    """Return a friendly Vietnamese label for a vehicle key."""
    if not vehicle_type:
        return ""
    if vehicle_type in VEHICLE_LABELS:
        return VEHICLE_LABELS[vehicle_type]
    alias = VEHICLE_ALIASES.get(vehicle_type)
    if alias and alias in VEHICLE_LABELS:
        return VEHICLE_LABELS[alias]
    return str(vehicle_type)


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


def load_penalty_data(path: str | Path = PENALTIES_PATH) -> dict[str, Any]:
    """Load penalty data from JSON."""
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


def _vehicle_record(entry: dict[str, Any], vehicle_type: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not vehicle_type or vehicle_type == "all":
        return None, None

    by_vehicle = entry.get("by_vehicle") or {}
    if vehicle_type in by_vehicle:
        record = dict(by_vehicle[vehicle_type])
        record["vehicle_name"] = get_vehicle_label(vehicle_type)
        return record, vehicle_type

    alias = VEHICLE_ALIASES.get(vehicle_type)
    if alias and alias in by_vehicle:
        record = dict(by_vehicle[alias])
        record["vehicle_name"] = f"{get_vehicle_label(vehicle_type)} (áp dụng nhóm {record.get('vehicle_name', alias)})"
        return record, alias
    if vehicle_type == "bus" and "car" in by_vehicle:
        record = dict(by_vehicle["car"])
        record["vehicle_name"] = f"{get_vehicle_label(vehicle_type)} (áp dụng nhóm {record.get('vehicle_name', 'Ô tô')})"
        return record, "car"

    return None, None


def get_penalties_for_sign(
    class_id,
    vehicle_type: str | None = None,
    signs_data: dict | None = None,
    penalties_data: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Return penalty records for a detected sign by class_id.

    The function is defensive: missing JSON files, missing refs, or unsupported
    vehicles return readable fallback records instead of raising.
    """
    sign_info = get_sign_info(class_id, signs_data)
    refs = sign_info.get("penalty_refs") or []
    if not refs:
        return [{"message": NO_PENALTY_DATA_MESSAGE, "sign_info": sign_info}]

    penalties = penalties_data if penalties_data is not None else load_penalty_data()
    records: list[dict[str, Any]] = []
    for ref in refs:
        entry = (penalties or {}).get(ref)
        if not entry:
            records.append(
                {
                    "penalty_key": ref,
                    "message": NO_PENALTY_DATA_MESSAGE,
                    "sign_info": sign_info,
                }
            )
            continue

        record: dict[str, Any] = {
            "penalty_key": ref,
            "behavior": entry.get("behavior", ""),
            "legal_document": entry.get("legal_document", ""),
            "effective_from": entry.get("effective_from", ""),
            "verified_at": entry.get("verified_at", ""),
            "note": entry.get("note", "Thông tin chỉ mang tính tham khảo phục vụ học tập."),
            "sign_info": sign_info,
        }

        if entry.get("calculation_required"):
            record["calculation_required"] = True
            record["message"] = entry.get(
                "note",
                "Cần biết tốc độ thực tế để xác định mức phạt tham khảo.",
            )
            records.append(record)
            continue

        if vehicle_type == "all":
            record["by_vehicle"] = entry.get("by_vehicle", {})
            records.append(record)
            continue

        vehicle_record, matched_vehicle_type = _vehicle_record(entry, vehicle_type)
        if vehicle_type and not vehicle_record:
            record["message"] = VEHICLE_NOT_APPLICABLE_MESSAGE
            record["vehicle_type"] = vehicle_type
            record["vehicle_name"] = get_vehicle_label(vehicle_type)
        elif vehicle_record:
            record["vehicle_type"] = vehicle_type
            record["matched_vehicle_type"] = matched_vehicle_type
            record["vehicle"] = vehicle_record
        else:
            record["by_vehicle"] = entry.get("by_vehicle", {})

        records.append(record)

    return records or [{"message": NO_PENALTY_DATA_MESSAGE, "sign_info": sign_info}]


def get_speed_penalty(
    speed_limit,
    actual_speed,
    vehicle_type: str,
    penalties_data: dict | None = None,
) -> dict[str, Any]:
    """
    Calculate speed over limit and return the matching penalty bracket.

    over_from <= over_speed < over_to; if over_to is None, over_speed >= over_from.
    """
    penalties = penalties_data if penalties_data is not None else load_penalty_data()
    entry = (penalties or {}).get("speed_limit")
    try:
        limit = float(speed_limit)
        actual = float(actual_speed)
    except (TypeError, ValueError):
        return {
            "message": "Không xác định được tốc độ để tính mức phạt tham khảo.",
            "legal_document": (entry or {}).get("legal_document", ""),
            "verified_at": (entry or {}).get("verified_at", ""),
        }

    over_speed = actual - limit
    base = {
        "penalty_key": "speed_limit",
        "behavior": (entry or {}).get("behavior", "Chạy quá tốc độ quy định ghi trên biển báo"),
        "legal_document": (entry or {}).get("legal_document", ""),
        "effective_from": (entry or {}).get("effective_from", ""),
        "verified_at": (entry or {}).get("verified_at", ""),
        "note": (entry or {}).get("note", "Thông tin chỉ mang tính tham khảo phục vụ học tập."),
        "speed_limit": limit,
        "actual_speed": actual,
        "over_speed": over_speed,
        "vehicle_type": vehicle_type,
    }

    if not entry:
        return {**base, "message": NO_PENALTY_DATA_MESSAGE}
    if vehicle_type == "all":
        return {
            **base,
            "message": "Hãy chọn một loại phương tiện cụ thể để tính mức phạt tốc độ.",
            "vehicle_name": get_vehicle_label(vehicle_type),
        }
    if over_speed < 5:
        return {
            **base,
            "message": "Tốc độ vượt dưới 5 km/h hoặc không vượt quá tốc độ, chưa thuộc các ngưỡng phạt được liệt kê trong dữ liệu.",
        }

    speed_rules = entry.get("speed_rules") or {}
    if vehicle_type == "bus":
        matched_vehicle_type = "car"
    else:
        matched_vehicle_type = vehicle_type if vehicle_type in speed_rules else VEHICLE_ALIASES.get(vehicle_type)
    rules = speed_rules.get(matched_vehicle_type or "")
    if not rules:
        return {
            **base,
            "message": VEHICLE_NOT_APPLICABLE_MESSAGE,
            "vehicle_name": get_vehicle_label(vehicle_type),
        }

    for rule in rules:
        over_from = float(rule.get("over_from", 0))
        over_to = rule.get("over_to")
        if over_speed >= over_from and (over_to is None or over_speed < float(over_to)):
            vehicle = dict(rule)
            vehicle["vehicle_name"] = get_vehicle_label(vehicle_type)
            return {
                **base,
                "matched_vehicle_type": matched_vehicle_type,
                "vehicle": vehicle,
            }

    return {
        **base,
        "message": "Chưa có khung mức phạt phù hợp với mức vượt tốc độ này trong dữ liệu.",
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
