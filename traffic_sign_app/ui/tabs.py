"""Streamlit tab renderers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from traffic_sign_app.config import AUDIO_DIR, DB_PATH, TEMP_DIR
from traffic_sign_app.core.detector import detect_image, draw_detections
from traffic_sign_app.core.warning_engine import generate_speech_text
from traffic_sign_app.services.chatbot import answer_question
from traffic_sign_app.services.knowledge_base import (
    clear_history,
    get_detection_history,
    get_sign_info,
)
from traffic_sign_app.services.reporting import export_history_csv, get_summary_stats
from traffic_sign_app.services.speech import should_speak, text_to_speech
from traffic_sign_app.ui.components import detect_and_render_image, show_sign_info, speak_sign
from traffic_sign_app.ui.state import save_detection_with_cooldown, sign_label


def render_image_tab(model, signs_data: dict, conf_threshold: float, enable_speech: bool) -> None:
    """Render image upload and detection workflow."""
    st.header("Nhận diện từ ảnh")
    uploaded_image = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not uploaded_image:
        return

    try:
        image = Image.open(uploaded_image).convert("RGB")
        image_rgb = np.array(image)
        col_a, col_b = st.columns(2)
        col_a.image(image_rgb, caption="Ảnh gốc", use_container_width=True)

        if model is None:
            st.info("Chưa thể detect vì model chưa sẵn sàng.")
            return

        if st.button("Detect ảnh", type="primary"):
            with st.spinner("Đang nhận diện biển báo..."):
                detections, annotated = detect_and_render_image(model, image_rgb, signs_data, conf_threshold)
            col_b.image(annotated, caption="Ảnh đã vẽ bounding box", use_container_width=True)

            if not detections:
                st.warning("Không phát hiện biển báo trong ảnh.")

            for idx, detection in enumerate(detections):
                sign_info = get_sign_info(detection["class_id"], signs_data)
                st.session_state.last_sign_info = sign_info
                save_detection_with_cooldown(detection, sign_info, "image", seconds=1)
                with st.expander(sign_info.get("class_name", "Biển báo"), expanded=True):
                    show_sign_info(sign_info, detection)
                    button_key = f"img_speak_{detection['class_id']}_{idx}"
                    if enable_speech and st.button("Đọc cảnh báo", key=button_key):
                        speak_sign(sign_info)
    except Exception as exc:
        st.error(f"Không xử lý được ảnh upload: {exc}")


def render_video_tab(
    model,
    signs_data: dict,
    conf_threshold: float,
    enable_speech: bool,
    video_stride: int,
) -> None:
    """Render video upload and frame-by-frame detection workflow."""
    st.header("Nhận diện từ video")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])
    max_frames = st.number_input("Giới hạn frame xử lý trong demo", min_value=30, max_value=3000, value=600, step=30)

    if not uploaded_video:
        return
    if model is None:
        st.info("Chưa thể detect video vì model chưa sẵn sàng.")
        return
    if not st.button("Xử lý video", type="primary"):
        return

    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
        tmp.write(uploaded_video.getbuffer())
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        st.error("Không mở được video upload.")
        return

    frame_placeholder = st.empty()
    result_placeholder = st.container()
    progress = st.progress(0)
    frame_index = 0
    last_seen: dict[int, dict] = {}
    last_audio_path = None

    while frame_index < int(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_index % int(video_stride) == 0:
            try:
                detections = detect_image(model, frame_bgr, conf_threshold)
                annotated = draw_detections(frame_bgr, detections, signs_data)
                frame_placeholder.image(
                    annotated,
                    channels="BGR",
                    caption=f"Frame {frame_index}",
                    use_container_width=True,
                )

                for detection in detections:
                    sign_info = get_sign_info(detection["class_id"], signs_data)
                    st.session_state.last_sign_info = sign_info
                    last_seen[int(detection["class_id"])] = {
                        "detection": detection,
                        "sign_info": sign_info,
                    }
                    save_detection_with_cooldown(detection, sign_info, "video", seconds=8)
                    if enable_speech and should_speak(detection["class_id"], cooldown_seconds=8):
                        last_audio_path = text_to_speech(generate_speech_text(sign_info), AUDIO_DIR)
            except Exception as exc:
                st.warning(f"Bỏ qua frame {frame_index} do lỗi inference: {exc}")

        frame_index += 1
        progress.progress(min(frame_index / int(max_frames), 1.0))

    cap.release()
    progress.empty()
    if not last_seen:
        st.warning("Không phát hiện biển báo trong video hoặc trong số frame đã xử lý.")
        return

    with result_placeholder:
        st.subheader("Biển mới phát hiện")
        for item in last_seen.values():
            with st.expander(item["sign_info"].get("class_name", "Biển báo")):
                show_sign_info(item["sign_info"], item["detection"])
        if enable_speech and last_audio_path:
            st.audio(last_audio_path)


def render_lookup_tab(signs_data: dict, enable_speech: bool) -> None:
    """Render sign lookup workflow."""
    st.header("Tra cứu biển báo")
    if not signs_data:
        st.error("Không tải được data/signs.json.")
        return

    selected = st.selectbox(
        "Chọn biển báo",
        options=sorted(signs_data.items(), key=lambda item: int(item[0])),
        format_func=sign_label,
    )
    selected_info = selected[1]
    st.session_state.last_sign_info = selected_info
    show_sign_info(selected_info)
    if enable_speech and st.button("Đọc cảnh báo tra cứu"):
        speak_sign(selected_info)


def render_chat_tab(signs_data: dict) -> None:
    """Render rule-based chatbot workflow."""
    st.header("Chatbot rule-based")
    current_info = st.session_state.last_sign_info
    if not current_info and signs_data:
        selected = st.selectbox(
            "Chưa có biển detect gần nhất. Chọn một biển để hỏi:",
            options=sorted(signs_data.items(), key=lambda item: int(item[0])),
            format_func=sign_label,
            key="chat_select_sign",
        )
        current_info = selected[1]

    if current_info:
        st.caption(f"Đang hỏi về: {current_info.get('class_name', 'Biển báo')}")

    question = st.chat_input("Hỏi: biển này ý nghĩa là gì, cần làm gì, phạt không, ví dụ...")
    if question:
        answer = answer_question(question, current_info)
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history[-12:]:
        with st.chat_message(role):
            st.markdown(message)


def render_quiz_tab(signs_data: dict, scenarios: dict) -> None:
    """Render quiz and learning scenario workflow."""
    st.header("Quiz / Tình huống học tập")
    if not scenarios:
        st.warning("Chưa có dữ liệu tình huống trong data/scenarios.json.")
        return

    scenario_items = sorted(scenarios.items(), key=lambda item: int(item[0]))
    selected = st.selectbox(
        "Chọn tình huống",
        scenario_items,
        format_func=lambda item: f"{item[0]} - {signs_data.get(item[0], {}).get('short_name', 'Tình huống')}",
    )
    class_id, scenario = selected
    st.info(scenario.get("scenario", ""))
    st.write(f"**Câu hỏi:** {scenario.get('question', '')}")
    choice = st.radio("Đáp án", scenario.get("options", []), key=f"quiz_{class_id}")
    if st.button("Kiểm tra đáp án"):
        if choice == scenario.get("answer"):
            st.success("Đúng rồi.")
        else:
            st.error(f"Chưa đúng. Đáp án đúng: {scenario.get('answer')}")
        st.write(scenario.get("explanation", ""))


def render_history_tab() -> None:
    """Render detection history, summary stats and CSV export."""
    st.header("Lịch sử / thống kê")
    history_df = get_detection_history(DB_PATH)
    stats = get_summary_stats(history_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng detection", stats["total"])
    c2.metric("Confidence TB", f"{stats['avg_confidence']:.3f}")
    c3.metric("Biển nhiều nhất", stats["top_sign"])
    c4.metric("Gần nhất", stats["latest_detection"])

    st.dataframe(history_df, use_container_width=True, hide_index=True)
    col_clear, col_export = st.columns(2)
    if col_clear.button("Clear history"):
        clear_history(DB_PATH)
        st.success("Đã xóa lịch sử.")
        st.rerun()
    if col_export.button("Export CSV"):
        csv_path = export_history_csv(history_df)
        st.success(f"Đã export CSV: {csv_path}")

