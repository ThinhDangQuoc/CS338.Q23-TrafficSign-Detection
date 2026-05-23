"""Reusable Streamlit UI components."""

from __future__ import annotations

import numpy as np
import streamlit as st

from traffic_sign_app.config import AUDIO_DIR, PENALTY_LEARNING_NOTE
from traffic_sign_app.core.detector import detect_image, draw_detections
from traffic_sign_app.core.warning_engine import (
    format_penalty_explanation,
    generate_full_explanation,
    generate_speech_text,
    generate_warning,
)
from traffic_sign_app.services.knowledge_base import get_penalties_for_sign, get_speed_penalty, get_vehicle_label
from traffic_sign_app.services.knowledge_base import get_sign_info


def apply_short_labels(detections: list[dict], signs_data: dict) -> list[dict]:
    """Mutate detection labels to be shorter for display, without changing IDs."""
    for detection in detections:
        sign_info = get_sign_info(detection.get("class_id"), signs_data)
        short_name = sign_info.get("short_name") or sign_info.get("class_name")
        if short_name:
            detection["class_name"] = short_name
    return detections
from traffic_sign_app.services.speech import text_to_speech


def _render_penalty_section(sign_info: dict, vehicle_type: str, key_prefix: str) -> None:
    penalty_refs = sign_info.get("penalty_refs") or []
    if not penalty_refs:
        return

    with st.container(border=True):
        st.markdown("#### Mức phạt tham khảo")
        st.caption(f"Đang xem theo phương tiện: {get_vehicle_label(vehicle_type)}")
        st.caption(PENALTY_LEARNING_NOTE)

        if sign_info.get("speed_limit_value") is not None:
            speed_limit = int(sign_info.get("speed_limit_value", 50))
            actual_speed = st.number_input(
                "Tốc độ thực tế của phương tiện (km/h)",
                min_value=0,
                max_value=200,
                value=speed_limit,
                key=f"{key_prefix}_actual_speed",
            )
            st.session_state.last_speed_values[str(sign_info.get("class_id"))] = int(actual_speed)
            penalty_record = get_speed_penalty(
                speed_limit=speed_limit,
                actual_speed=actual_speed,
                vehicle_type=vehicle_type,
            )
            st.markdown(format_penalty_explanation(penalty_record, vehicle_type))
            return

        penalty_records = get_penalties_for_sign(sign_info.get("class_id"), vehicle_type=vehicle_type)
        st.markdown(format_penalty_explanation(penalty_records, vehicle_type))


def show_sign_info(
    sign_info: dict,
    detection: dict | None = None,
    vehicle_type: str = "car",
    key_prefix: str | None = None,
) -> None:
    """Render the full explanation block for one sign."""
    key_prefix = key_prefix or f"sign_{sign_info.get('class_id', 'unknown')}"

    st.subheader(sign_info.get("class_name", "Biển báo"))
    if detection:
        c1, c2, c3 = st.columns(3)
        c1.metric("Class ID", detection.get("class_id", sign_info.get("class_id", "")))
        c2.metric("Confidence", f"{float(detection.get('confidence', 0.0)):.2f}")
        c3.metric("Loại biển", sign_info.get("type", "Chưa rõ"))
    else:
        c1, c2 = st.columns(2)
        c1.metric("Class ID", sign_info.get("class_id", ""))
        c2.metric("Loại biển", sign_info.get("type", "Chưa rõ"))

    with st.container(border=True):
        st.markdown("#### Cảnh báo và giải thích")
        st.info(generate_warning(sign_info))
        st.markdown(generate_full_explanation(sign_info))
    _render_penalty_section(sign_info, vehicle_type, key_prefix)


def render_explanation_card(sign_info: dict) -> None:
    """Render a focused explanation card for the selected sign."""
    with st.container(border=True):
        st.markdown("#### Giải thích biển báo")
        st.info(generate_warning(sign_info))
        st.markdown(generate_full_explanation(sign_info))


def render_penalty_card(sign_info: dict, vehicle_type: str, key_prefix: str) -> None:
    """Render a focused penalty card for the selected sign."""
    _render_penalty_section(sign_info, vehicle_type, key_prefix)


def speak_sign(sign_info: dict) -> None:
    """Create and play a speech warning for one sign."""
    audio_path = text_to_speech(generate_speech_text(sign_info), AUDIO_DIR)
    if audio_path:
        st.audio(audio_path)
    else:
        st.warning("Không tạo được audio TTS. App vẫn tiếp tục chạy bình thường.")


def detect_and_render_image(
    model,
    image_rgb: np.ndarray,
    signs_data: dict,
    conf_threshold: float,
    img_size: int | None = None,
):
    """Detect signs and draw annotations for an image/frame."""
    detections = detect_image(model, image_rgb, conf_threshold, img_size)
    apply_short_labels(detections, signs_data)
    annotated = draw_detections(image_rgb, detections, signs_data)
    return detections, annotated
