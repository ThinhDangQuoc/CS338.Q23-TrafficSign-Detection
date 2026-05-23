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
    render_model_evaluation_tab,
    render_quiz_tab,
    render_video_tab,
    render_webcam_tab,
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
        @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
        html, body, .stApp {
            font-family: "Be Vietnam Pro", "Noto Sans", "DejaVu Sans", sans-serif;
        }
        .material-icons, .material-symbols-outlined {
            font-family: "Material Icons", "Material Symbols Outlined" !important;
        }
        body {
            background: radial-gradient(1200px 520px at 10% -10%, rgba(14, 165, 233, 0.16), transparent 60%),
                        radial-gradient(900px 380px at 90% 10%, rgba(124, 58, 237, 0.12), transparent 55%),
                        #0b1220;
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
        }
        .app-hero {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(135deg, rgba(10, 18, 32, 0.98), rgba(18, 28, 46, 0.96));
            border-radius: 14px;
            padding: 1rem 1.35rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.28);
        }
        .app-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.85rem;
            line-height: 1.22;
            letter-spacing: -0.01em;
            color: #f8fafc;
        }
        .app-hero p {
            margin: 0;
            color: #94a3b8;
            max-width: 880px;
            font-size: 0.95rem;
        }
        .legal-note {
            border-left: 3px solid #38bdf8;
            color: #cbd5e1;
            background: rgba(9, 16, 30, 0.68);
            padding: 0.65rem 0.85rem;
            border-radius: 8px;
            margin: 0.6rem 0 1rem 0;
            font-size: 0.88rem;
        }
        .sidebar-group {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(9, 16, 30, 0.7);
            border-radius: 12px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-title {
            color: #e2e8f0;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .sidebar-item {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            color: #cbd5e1;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            font-size: 0.88rem;
        }
        .sidebar-item:last-child {
            border-bottom: none;
        }
        .sidebar-item span {
            color: #f8fafc;
            font-weight: 600;
        }
        .section-card {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(9, 16, 30, 0.75);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 16px 32px rgba(2, 6, 23, 0.35);
        }
        .stepper {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.7rem 0 1rem 0;
        }
        .step {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 0.75rem 0.8rem;
            background: rgba(9, 16, 30, 0.6);
            color: #cbd5e1;
            font-size: 0.9rem;
        }
        .step strong {
            color: #67e8f9;
            font-weight: 600;
        }
        .panel-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.45rem;
        }
        .result-table {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(9, 16, 30, 0.65);
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
        }
        .empty-state {
            border: 1px dashed rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            color: #94a3b8;
            background: rgba(9, 16, 30, 0.6);
        }
        div[data-testid="stMetric"] {
            background: rgba(9, 16, 30, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 0.75rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #67e8f9 !important;
            border-bottom: 2px solid #22d3ee !important;
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        f"""
        <section class="app-hero">
            <h1>Hệ thống học tập nhận diện biển báo giao thông Việt Nam</h1>
            <p>Tải ảnh hoặc video để AI nhận diện biển báo, giải thích ý nghĩa và hỗ trợ học tập.</p>
        </section>
        <div class="legal-note">{PENALTY_LEARNING_NOTE}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(classes: list[str], signs_data: dict, model_ready: bool) -> tuple[float, int, int, bool, bool, bool, str]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-group">
                <div class="sidebar-title">Trạng thái hệ thống</div>
                <div class="sidebar-item">Model<span>{"Sẵn sàng" if model_ready else "Chưa tải"}</span></div>
                <div class="sidebar-item">Dữ liệu<span>{len(signs_data)} biển báo</span></div>
                <div class="sidebar-item">Database<span>SQLite ready</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-group'><div class='sidebar-title'>Thiết lập nhận diện</div>", unsafe_allow_html=True)
        conf_threshold = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
        img_size = st.selectbox("Image size / inference size", options=[416, 640, 960], index=1)
        if st.session_state.get("selected_vehicle_type") not in VEHICLE_OPTIONS:
            st.session_state.selected_vehicle_type = "car"
        selected_vehicle_type = st.selectbox(
            "Loại phương tiện",
            options=list(VEHICLE_OPTIONS),
            format_func=lambda key: VEHICLE_OPTIONS[key],
            key="selected_vehicle_type",
        )
        video_stride = st.number_input("Process every N frames", min_value=1, max_value=30, value=5)
        save_history = st.toggle("Lưu lịch sử nhận diện", value=True)
        enable_speech = st.toggle("Đọc cảnh báo bằng giọng nói", value=True)
        show_performance = st.toggle("Hiển thị FPS / inference time", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return (
        conf_threshold,
        int(img_size),
        int(video_stride),
        enable_speech,
        save_history,
        show_performance,
        selected_vehicle_type,
    )


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
    _render_header()
    (
        conf_threshold,
        img_size,
        video_stride,
        enable_speech,
        save_history,
        show_performance,
        selected_vehicle_type,
    ) = _render_sidebar(
        classes,
        signs_data,
        model is not None,
    )

    tab_image, tab_video, tab_webcam, tab_lookup, tab_chat, tab_quiz, tab_history, tab_eval = st.tabs(
        [
            "Ảnh",
            "Video",
            "Webcam",
            "Tra cứu",
            "Chatbot",
            "Quiz",
            "Lịch sử",
            "Đánh giá model",
        ]
    )

    with tab_image:
        render_image_tab(
            model,
            signs_data,
            conf_threshold,
            img_size,
            enable_speech,
            save_history,
            show_performance,
            selected_vehicle_type,
        )
    with tab_video:
        render_video_tab(
            model,
            signs_data,
            conf_threshold,
            img_size,
            enable_speech,
            save_history,
            show_performance,
            video_stride,
            selected_vehicle_type,
        )
    with tab_webcam:
        render_webcam_tab(
            model,
            signs_data,
            conf_threshold,
            img_size,
            enable_speech,
            save_history,
            show_performance,
            video_stride,
            selected_vehicle_type,
        )
    with tab_lookup:
        render_lookup_tab(signs_data, enable_speech, selected_vehicle_type)
    with tab_chat:
        render_chat_tab(signs_data, selected_vehicle_type)
    with tab_quiz:
        render_quiz_tab(signs_data, scenarios)
    with tab_history:
        render_history_tab()
    with tab_eval:
        render_model_evaluation_tab(model, classes)
