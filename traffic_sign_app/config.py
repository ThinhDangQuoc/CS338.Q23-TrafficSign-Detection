"""Centralized paths and app constants."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
DB_PATH = BASE_DIR / "database" / "app.db"
AUDIO_DIR = BASE_DIR / "assets" / "audio"
TEMP_DIR = BASE_DIR / "assets" / "temp"
CACHE_DIR = BASE_DIR / ".cache"

CLASSES_PATH = DATA_DIR / "classes.txt"
SIGNS_PATH = DATA_DIR / "signs.json"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
PENALTIES_PATH = DATA_DIR / "penalties.json"

PENALTY_LEARNING_NOTE = (
    "Thông tin mức phạt chỉ mang tính tham khảo phục vụ học tập. "
    "Mức phạt thực tế phụ thuộc loại phương tiện, hành vi cụ thể, "
    "tình tiết vi phạm và quy định pháp luật hiện hành."
)
