"""Main Streamlit application composition."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from traffic_sign_app.config import (
    CLASSES_PATH,
    DB_PATH,
    MODEL_PATH,
    PENALTY_LEARNING_NOTE,
    SCENARIOS_PATH,
    SIGNS_PATH,
    TEMP_DIR,
)
from traffic_sign_app.core.detector import load_model
from traffic_sign_app.services.knowledge_base import init_db, load_classes, load_sign_data
from traffic_sign_app.services.knowledge_base import VEHICLE_LABELS
from traffic_sign_app.ui.state import ensure_state
from traffic_sign_app.ui.tabs import (
    render_chat_tab,
    render_history_tab,
    render_image_tab,
    render_lookup_tab,
    render_quiz_tab,
    render_video_tab,
)


VEHICLE_OPTIONS = {key: VEHICLE_LABELS[key] for key in ("car", "motorbike", "truck", "bus", "all")}


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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }
        .app-hero {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(31, 41, 55, 0.96));
            border-radius: 10px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
        }
        .app-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            line-height: 1.18;
            letter-spacing: 0;
            color: #f8fafc;
        }
        .app-hero p {
            margin: 0;
            color: #cbd5e1;
            max-width: 860px;
            font-size: 0.98rem;
        }
        .status-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }
        .status-badge {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            color: #e2e8f0;
            background: rgba(15, 23, 42, 0.72);
            font-size: 0.86rem;
        }
        .status-badge.ready {
            border-color: rgba(34, 197, 94, 0.5);
            color: #bbf7d0;
        }
        .status-badge.warn {
            border-color: rgba(245, 158, 11, 0.55);
            color: #fde68a;
        }
        .legal-note {
            border-left: 3px solid #38bdf8;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.62);
            padding: 0.72rem 0.9rem;
            border-radius: 6px;
            margin: 0.55rem 0 1rem 0;
            font-size: 0.9rem;
        }
        .sidebar-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(15, 23, 42, 0.62);
            border-radius: 8px;
            padding: 0.72rem 0.78rem;
            margin: 0.48rem 0;
        }
        .sidebar-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.18rem;
        }
        .sidebar-value {
            color: #f8fafc;
            font-size: 0.98rem;
            font-weight: 650;
        }
        .soft-panel {
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(15, 23, 42, 0.44);
            border-radius: 8px;
            padding: 1rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(model_ready: bool, signs_count: int) -> None:
    model_badge_class = "ready" if model_ready else "warn"
    model_badge = "YOLO model ready" if model_ready else "YOLO model chưa tải"
    st.markdown(
        f"""
        <section class="app-hero">
            <h1>Hệ thống học tập nhận diện biển báo giao thông Việt Nam</h1>
            <p>AI demo dùng YOLO để nhận diện biển báo, giải thích ý nghĩa, cảnh báo hành vi và hiển thị mức phạt tham khảo phục vụ học tập.</p>
            <div class="status-row">
                <span class="status-badge {model_badge_class}">{model_badge}</span>
                <span class="status-badge ready">SQLite ready</span>
                <span class="status-badge">{signs_count} biển báo</span>
            </div>
        </section>
        <div class="legal-note">{PENALTY_LEARNING_NOTE}</div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">{label}</div>
            <div class="sidebar-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(classes: list[str], signs_data: dict, model_ready: bool) -> tuple[float, bool, int, str]:
    with st.sidebar:
        st.header("Điều khiển demo")
        _sidebar_card("Model", "YOLO11s Traffic Sign")
        _sidebar_card("Trạng thái model", "Đã tải thành công" if model_ready else "Chưa tải")
        _sidebar_card("Số class", f"{len(classes)} class" if classes else "Chưa đọc được classes.txt")
        _sidebar_card("Số biển trong knowledge base", f"{len(signs_data)} biển" if signs_data else "Chưa đọc được signs.json")
        _sidebar_card("Database", "SQLite ready")

        st.divider()
        st.subheader("Thiết lập nhận diện")
        conf_threshold = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
        enable_speech = st.toggle("Đọc cảnh báo bằng giọng nói", value=True)
        video_stride = st.number_input("Detect mỗi N frame video", min_value=1, max_value=30, value=5)
        if st.session_state.get("selected_vehicle_type") not in VEHICLE_OPTIONS:
            st.session_state.selected_vehicle_type = "car"
        selected_vehicle_type = st.selectbox(
            "Loại phương tiện",
            options=list(VEHICLE_OPTIONS),
            format_func=lambda key: VEHICLE_OPTIONS[key],
            key="selected_vehicle_type",
        )

    return conf_threshold, enable_speech, int(video_stride), selected_vehicle_type


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
    _inject_styles()
    ensure_state()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)

    classes = cached_classes()
    signs_data = cached_signs()
    scenarios = cached_scenarios()

    model_path = str(MODEL_PATH)
    model = _load_model_or_show_warning(model_path)
    _render_header(model is not None, len(signs_data))
    conf_threshold, enable_speech, video_stride, selected_vehicle_type = _render_sidebar(
        classes,
        signs_data,
        model is not None,
    )

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
        render_image_tab(model, signs_data, conf_threshold, enable_speech, selected_vehicle_type)
    with tab_video:
        render_video_tab(model, signs_data, conf_threshold, enable_speech, video_stride, selected_vehicle_type)
    with tab_lookup:
        render_lookup_tab(signs_data, enable_speech, selected_vehicle_type)
    with tab_chat:
        render_chat_tab(signs_data, selected_vehicle_type)
    with tab_quiz:
        render_quiz_tab(signs_data, scenarios)
    with tab_history:
        render_history_tab()
