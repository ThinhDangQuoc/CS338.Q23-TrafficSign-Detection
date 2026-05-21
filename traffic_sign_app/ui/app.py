"""Main Streamlit application composition."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from traffic_sign_app.config import (
    CLASSES_PATH,
    DB_PATH,
    MODEL_PATH,
    SCENARIOS_PATH,
    SIGNS_PATH,
    TEMP_DIR,
)
from traffic_sign_app.core.detector import load_model
from traffic_sign_app.services.knowledge_base import init_db, load_classes, load_sign_data
from traffic_sign_app.ui.state import ensure_state
from traffic_sign_app.ui.tabs import (
    render_chat_tab,
    render_history_tab,
    render_image_tab,
    render_lookup_tab,
    render_quiz_tab,
    render_video_tab,
)


@st.cache_data(show_spinner=False)
def cached_classes() -> list[str]:
    return load_classes(CLASSES_PATH)


@st.cache_data(show_spinner=False)
def cached_signs() -> dict:
    return load_sign_data(SIGNS_PATH)


@st.cache_data(show_spinner=False)
def cached_scenarios() -> dict:
    if not SCENARIOS_PATH.exists():
        return {}
    try:
        return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    return load_model(model_path)


def _render_sidebar(classes: list[str], signs_data: dict) -> tuple[str, float, bool, int]:
    with st.sidebar:
        st.header("Cấu hình")
        model_path = st.text_input("Đường dẫn model", value=str(MODEL_PATH))
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        enable_speech = st.toggle("Bật speech", value=True)
        video_stride = st.number_input("Detect mỗi N frame video", min_value=1, max_value=30, value=5)

        st.divider()
        st.subheader("Trạng thái")
        st.write("Model:", "✅ Có" if Path(model_path).exists() else "⚠️ Thiếu")
        st.write("classes.txt:", f"✅ {len(classes)} class" if classes else "⚠️ Thiếu hoặc lỗi")
        st.write("signs.json:", f"✅ {len(signs_data)} biển" if signs_data else "⚠️ Thiếu hoặc lỗi")
        st.write("SQLite:", "✅ Sẵn sàng")

    return model_path, conf_threshold, enable_speech, int(video_stride)


def _load_model_or_show_warning(model_path: str):
    if not Path(model_path).exists():
        st.warning("Thiếu model. Hãy đặt file best.pt vào models/best.pt hoặc nhập đúng đường dẫn ở sidebar.")
        return None

    try:
        return cached_model(model_path)
    except Exception as exc:
        st.warning(str(exc))
        return None


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Traffic Sign Learning System",
        page_icon="🚦",
        layout="wide",
    )
    ensure_state()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)

    classes = cached_classes()
    signs_data = cached_signs()
    scenarios = cached_scenarios()

    st.title("Hệ thống học tập nhận diện biển báo giao thông Việt Nam")
    st.caption(
        "Hệ thống hỗ trợ học tập, nhận diện và giải thích biển báo giao thông trong môi trường mô phỏng, ảnh, video hoặc camera."
    )

    model_path, conf_threshold, enable_speech, video_stride = _render_sidebar(classes, signs_data)
    model = _load_model_or_show_warning(model_path)

    tab_image, tab_video, tab_lookup, tab_chat, tab_quiz, tab_history = st.tabs(
        [
            "Nhận diện ảnh",
            "Nhận diện video",
            "Tra cứu biển báo",
            "Chatbot hỏi đáp",
            "Quiz / Tình huống",
            "Lịch sử / thống kê",
        ]
    )

    with tab_image:
        render_image_tab(model, signs_data, conf_threshold, enable_speech)
    with tab_video:
        render_video_tab(model, signs_data, conf_threshold, enable_speech, video_stride)
    with tab_lookup:
        render_lookup_tab(signs_data, enable_speech)
    with tab_chat:
        render_chat_tab(signs_data)
    with tab_quiz:
        render_quiz_tab(signs_data, scenarios)
    with tab_history:
        render_history_tab()
