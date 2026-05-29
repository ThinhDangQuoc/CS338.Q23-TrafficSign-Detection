"""Session-state helpers for the Streamlit app."""

from __future__ import annotations

import time

import streamlit as st

from traffic_sign_app.config import DB_PATH
from traffic_sign_app.services.knowledge_base import save_detection


def ensure_state() -> None:
    """Initialize keys used across tabs."""
    st.session_state.setdefault("last_sign_info", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("saved_recently", {})
    st.session_state.setdefault("selected_vehicle_type", "car")
    st.session_state.setdefault("last_speed_values", {})


def sign_label(item: tuple[str, dict]) -> str:
    """Format a sign option for Streamlit selectbox."""
    class_id, info = item
    return f"{class_id} - {info.get('class_name', 'Chưa rõ')}"


def save_detection_with_cooldown(
    detection: dict,
    sign_info: dict,
    source: str,
    seconds: int = 5,
    enabled: bool = True,
) -> None:
    """Save a detection while avoiding repeated rows for the same class/source or track_id."""
    if not enabled:
        return

    track_id = detection.get("track_id")
    if track_id is not None:
        key = f"{source}:track:{track_id}"
    else:
        key = f"{source}:{detection.get('class_id')}"

    now = time.time()
    last_saved = st.session_state.saved_recently.get(key, 0)
    if now - last_saved < seconds:
        return

    save_detection(
        detection.get("class_id", -1),
        sign_info.get("class_name", detection.get("class_name", "Unknown")),
        float(detection.get("confidence", 0.0)),
        sign_info.get("meaning", ""),
        source,
        DB_PATH,
    )
    st.session_state.saved_recently[key] = now
