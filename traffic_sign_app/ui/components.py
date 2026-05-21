"""Reusable Streamlit UI components."""

from __future__ import annotations

import numpy as np
import streamlit as st

from traffic_sign_app.config import AUDIO_DIR
from traffic_sign_app.core.detector import detect_image, draw_detections
from traffic_sign_app.core.warning_engine import (
    generate_full_explanation,
    generate_speech_text,
    generate_warning,
)
from traffic_sign_app.services.speech import text_to_speech


def show_sign_info(sign_info: dict, detection: dict | None = None) -> None:
    """Render the full explanation block for one sign."""
    if detection:
        c1, c2, c3 = st.columns(3)
        c1.metric("Class ID", detection.get("class_id", sign_info.get("class_id", "")))
        c2.metric("Confidence", f"{float(detection.get('confidence', 0.0)):.2f}")
        c3.metric("Loại biển", sign_info.get("type", "Chưa rõ"))
    else:
        st.caption(f"Class ID: {sign_info.get('class_id')} | Loại: {sign_info.get('type', 'Chưa rõ')}")

    st.subheader(sign_info.get("class_name", "Biển báo"))
    st.info(generate_warning(sign_info))
    st.markdown(generate_full_explanation(sign_info))


def speak_sign(sign_info: dict) -> None:
    """Create and play a speech warning for one sign."""
    audio_path = text_to_speech(generate_speech_text(sign_info), AUDIO_DIR)
    if audio_path:
        st.audio(audio_path)
    else:
        st.warning("Không tạo được audio TTS. App vẫn tiếp tục chạy bình thường.")


def detect_and_render_image(model, image_rgb: np.ndarray, signs_data: dict, conf_threshold: float):
    """Detect signs and draw annotations for an image/frame."""
    detections = detect_image(model, image_rgb, conf_threshold)
    annotated = draw_detections(image_rgb, detections, signs_data)
    return detections, annotated

