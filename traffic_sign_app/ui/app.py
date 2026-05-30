"""Main Streamlit application composition."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from traffic_sign_app.config import (
    BASE_DIR,
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
            padding-top: 0.85rem;
            padding-bottom: 2.2rem;
        }
        .app-hero {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(135deg, rgba(10, 18, 32, 0.98), rgba(18, 28, 46, 0.96));
            border-radius: 14px;
            padding: 0.85rem 1.2rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.28);
        }
        .app-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.7rem;
            line-height: 1.22;
            letter-spacing: -0.01em;
            color: #f8fafc;
        }
        .app-hero p {
            margin: 0;
            color: #94a3b8;
            max-width: 880px;
            font-size: 0.92rem;
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
        .section-title {
            color: #e2e8f0;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.2rem 0;
        }
        .section-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 0.7rem;
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
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            background: rgba(9, 16, 30, 0.75) !important;
            border-radius: 14px !important;
            padding: 0.85rem !important;
            box-shadow: 0 16px 32px rgba(2, 6, 23, 0.35) !important;
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
        .step--done {
            border-color: rgba(34, 211, 238, 0.6);
            background: rgba(34, 211, 238, 0.08);
            color: #e2e8f0;
        }
        .step--active {
            border-color: rgba(99, 102, 241, 0.55);
            background: rgba(99, 102, 241, 0.12);
            color: #f8fafc;
        }
        .step--warning {
            border-color: rgba(251, 191, 36, 0.6);
            background: rgba(251, 191, 36, 0.1);
            color: #fde68a;
        }
        .step--error {
            border-color: rgba(248, 113, 113, 0.65);
            background: rgba(248, 113, 113, 0.08);
            color: #fecaca;
        }
        .result-highlight {
            border: 1px solid rgba(34, 211, 238, 0.55);
            box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.25), 0 12px 30px rgba(15, 23, 42, 0.35);
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
        .panel-subtitle {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 0.6rem;
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
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.4rem;
        }
        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 0.75rem;
            background: rgba(8, 15, 28, 0.6);
        }

        /* ═══ STREAMLIT ELEMENT CONTAINER MARKER SELECTORS ═══ */

        /* ── Main Navigation (st.segmented_control with key="main_section") ── */
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] [role="radiogroup"] {
            gap: 1.25rem !important;
            justify-content: center !important;
            display: flex !important;
        }
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button {
            padding: 1.8rem 4.5rem !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            min-height: 90px !important;
            border-radius: 999px !important;
            letter-spacing: 0.02em !important;
            border: 2.5px solid rgba(148, 163, 184, 0.28) !important;
            background-color: rgba(15, 23, 42, 0.6) !important;
            color: #cbd5e1 !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button:hover {
            border-color: rgba(34, 211, 238, 0.7) !important;
            color: #ffffff !important;
            background-color: rgba(34, 211, 238, 0.1) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 24px rgba(34, 211, 238, 0.15) !important;
        }
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[data-selected="true"],
        [data-testid="stElementContainer"]:has(#main-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button.e8vg11g13 {
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.25), rgba(99, 102, 241, 0.22)) !important;
            border: 2.5px solid rgba(34, 211, 238, 0.95) !important;
            color: #ffffff !important;
            box-shadow:
                0 0 0 3px rgba(34, 211, 238, 0.3),
                0 10px 35px rgba(34, 211, 238, 0.3),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
            text-shadow: 0 0 20px rgba(34, 211, 238, 0.8) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Sub Navigation (key="detect_tab", "learn_tab", "track_tab") ── */
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] [role="radiogroup"],
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] [role="radiogroup"],
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] [role="radiogroup"] {
            gap: 0.8rem !important;
            justify-content: center !important;
            display: flex !important;
        }
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button,
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button,
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button {
            border: 1.5px solid rgba(148, 163, 184, 0.25) !important;
            background-color: rgba(15, 23, 42, 0.65) !important;
            padding: 0.8rem 1.8rem !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
            color: #94a3b8 !important;
            min-height: 52px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            letter-spacing: 0.01em !important;
            transition: all 0.18s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button:hover,
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button:hover,
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button:hover {
            border-color: rgba(34, 211, 238, 0.6) !important;
            color: #e2e8f0 !important;
            background-color: rgba(34, 211, 238, 0.08) !important;
        }
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[data-selected="true"],
        [data-testid="stElementContainer"]:has(#detect-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button.e8vg11g13,
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[data-selected="true"],
        [data-testid="stElementContainer"]:has(#learn-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button.e8vg11g13,
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button[data-selected="true"],
        [data-testid="stElementContainer"]:has(#track-tabs-marker) + [data-testid="stElementContainer"] [data-testid="stSegmentedControl"] button.e8vg11g13 {
            border: 1.5px solid rgba(34, 211, 238, 0.85) !important;
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(99, 102, 241, 0.13)) !important;
            color: #e6fbff !important;
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.28), 0 4px 20px rgba(34, 211, 238, 0.18) !important;
        }

        /* Divider between main nav and sub-nav */
        .sub-nav-wrapper {
            margin-top: 0.6rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
        }
        .nav-block {
            margin: 0.2rem 0 0rem 0;
        }
        .stButton > button {
            border-radius: 10px;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            border: none;
            color: #0b1220;
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


def _render_sidebar(
    classes: list[str],
    signs_data: dict,
    model_ready: bool,
    model_options: list[str],
) -> tuple[float, int, int, bool, bool, bool, str]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-group">
                <div class="sidebar-title">Trạng thái hệ thống</div>
                <div class="sidebar-item">Model<span>{"Sẵn sàng" if model_ready else "Chưa tải"}</span></div>
                <div class="sidebar-item">Dữ liệu<span>{len(signs_data)} biển báo</span></div>
                <div class="sidebar-item">Database<span>Sẵn sàng</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("<div class='sidebar-title'>Mô hình AI</div>", unsafe_allow_html=True)
            
            # Match current selection index or default to 0
            curr_val = st.session_state.get("selected_model_path")
            def_idx = 0
            if curr_val in model_options:
                def_idx = model_options.index(curr_val)
                
            st.selectbox(
                "Chọn file model",
                options=model_options,
                index=def_idx,
                key="selected_model_path",
            )
            
            st.markdown("<div class='sidebar-title'>Thiết lập nhận diện</div>", unsafe_allow_html=True)
            conf_threshold = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
            if st.session_state.get("selected_vehicle_type") not in VEHICLE_OPTIONS:
                st.session_state.selected_vehicle_type = "car"
            selected_vehicle_type = st.selectbox(
                "Loại phương tiện",
                options=list(VEHICLE_OPTIONS),
                format_func=lambda key: VEHICLE_OPTIONS[key],
                key="selected_vehicle_type",
            )
            save_history = st.toggle("Lưu lịch sử nhận diện", value=True)
            enable_speech = st.toggle("Đọc cảnh báo bằng giọng nói", value=True)
            with st.expander("Cài đặt nâng cao", expanded=False):
                img_size = st.selectbox("Kích thước xử lý ảnh", options=[416, 640, 960], index=1)
                video_stride = st.number_input("Xử lý mỗi N khung hình", min_value=1, max_value=30, value=5)
                show_performance = st.toggle("Hiển thị tốc độ xử lý", value=True)

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
    if not model_path:
        return None
    path = Path(model_path)
    if not path.is_absolute():
        from traffic_sign_app.config import BASE_DIR
        path = BASE_DIR / path

    if not path.exists():
        st.warning(f"Thiếu model tại {path.name}. Hãy đặt file model vào thư mục models/.")
        return None

    try:
        return cached_model(str(path))
    except Exception as exc:
        st.warning(f"Lỗi khi load model: {exc}")
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

    # Find available models in base_dir/models
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    model_files = sorted(list(models_dir.glob("*.pt")) + list(models_dir.glob("*.onnx")), key=lambda p: p.name)
    model_options = [str(p.relative_to(BASE_DIR).as_posix()) for p in model_files]
    if not model_options:
        model_options = ["models/best.pt"]

    # Set default model path if not initialized
    if "selected_model_path" not in st.session_state or st.session_state.selected_model_path not in model_options:
        st.session_state.selected_model_path = model_options[0]

    model_path = st.session_state.selected_model_path
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
        model_options,
    )

    # ── Main navigation ─────────────────────────────────────────────────────
    st.markdown("<div class='nav-block'></div>", unsafe_allow_html=True)
    st.markdown("<div id='main-tabs-marker'></div>", unsafe_allow_html=True)
    main_section = st.segmented_control(
        "Nhóm chức năng",
        ["🚦 Nhận diện", "📖 Học & Luyện tập", "📊 Theo dõi"],
        default=st.session_state.get("main_section", "🚦 Nhận diện"),
        key="main_section",
        label_visibility="collapsed",
    )

    # ── Sub-navigation & content ─────────────────────────────────────────────
    if main_section == "🚦 Nhận diện":
        st.markdown("<div class='sub-nav-wrapper'></div>", unsafe_allow_html=True)
        st.markdown("<div id='detect-tabs-marker'></div>", unsafe_allow_html=True)
        sub_tab = st.segmented_control(
            "Chọn chế độ nhận diện",
            ["🖼️ Tải ảnh", "🎬 Video", "🎥 Realtime"],
            default=st.session_state.get("detect_tab", "🖼️ Tải ảnh"),
            key="detect_tab",
            label_visibility="collapsed",
        )
        if sub_tab == "🖼️ Tải ảnh":
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
        elif sub_tab == "🎬 Video":
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
        else:
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
    elif main_section == "📖 Học & Luyện tập":
        st.markdown("<div class='sub-nav-wrapper'></div>", unsafe_allow_html=True)
        st.markdown("<div id='learn-tabs-marker'></div>", unsafe_allow_html=True)
        sub_tab = st.segmented_control(
            "Chọn chế độ học",
            ["🔍 Tra cứu", "🤖 Hỏi AI", "🧠 Quiz"],
            default=st.session_state.get("learn_tab", "🔍 Tra cứu"),
            key="learn_tab",
            label_visibility="collapsed",
        )
        if sub_tab == "🔍 Tra cứu":
            render_lookup_tab(signs_data, enable_speech, selected_vehicle_type)
        elif sub_tab == "🤖 Hỏi AI":
            render_chat_tab(signs_data, selected_vehicle_type)
        else:
            render_quiz_tab(signs_data, scenarios)
    else:
        st.markdown("<div class='sub-nav-wrapper'></div>", unsafe_allow_html=True)
        st.markdown("<div id='track-tabs-marker'></div>", unsafe_allow_html=True)
        sub_tab = st.segmented_control(
            "Chọn mục theo dõi",
            ["📋 Lịch sử", "🔬 Đánh giá model"],
            default=st.session_state.get("track_tab", "📋 Lịch sử"),
            key="track_tab",
            label_visibility="collapsed",
        )
        if sub_tab == "📋 Lịch sử":
            render_history_tab()
        else:
            render_model_evaluation_tab(model, classes)
