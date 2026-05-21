"""Backward-compatible imports for data and SQLite helpers."""

from traffic_sign_app.services.knowledge_base import (
    clear_history,
    get_detection_history,
    get_sign_info,
    init_db,
    load_classes,
    load_sign_data,
    save_detection,
)

__all__ = [
    "load_classes",
    "load_sign_data",
    "get_sign_info",
    "init_db",
    "save_detection",
    "get_detection_history",
    "clear_history",
]

