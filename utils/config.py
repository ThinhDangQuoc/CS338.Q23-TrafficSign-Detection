"""Backward-compatible imports for shared app paths."""

from traffic_sign_app.config import (
    AUDIO_DIR,
    BASE_DIR,
    CACHE_DIR,
    CLASSES_PATH,
    DATA_DIR,
    DB_PATH,
    MODEL_PATH,
    SCENARIOS_PATH,
    SIGNS_PATH,
    TEMP_DIR,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODEL_PATH",
    "DB_PATH",
    "AUDIO_DIR",
    "TEMP_DIR",
    "CACHE_DIR",
    "CLASSES_PATH",
    "SIGNS_PATH",
    "SCENARIOS_PATH",
]

