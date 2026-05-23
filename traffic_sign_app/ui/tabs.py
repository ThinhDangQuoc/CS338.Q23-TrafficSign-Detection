"""Streamlit tab renderers."""

from __future__ import annotations

import threading
import time
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from traffic_sign_app.config import AUDIO_DIR, BASE_DIR, DB_PATH, MODEL_PATH, TEMP_DIR
from traffic_sign_app.core.detector import detect_image, draw_detections
from traffic_sign_app.core.warning_engine import generate_speech_text, generate_warning
from traffic_sign_app.services.chatbot import answer_question
from traffic_sign_app.services.knowledge_base import (
    clear_history,
    get_detection_history,
    get_sign_info,
    get_vehicle_label,
    save_detection,
)
from traffic_sign_app.services.reporting import export_history_csv, get_summary_stats
from traffic_sign_app.services.speech import should_speak, text_to_speech
from traffic_sign_app.ui.components import (
    apply_short_labels,
    detect_and_render_image,
    render_explanation_card,
    render_penalty_card,
    show_sign_info,
    speak_sign,
)
from traffic_sign_app.ui.state import save_detection_with_cooldown, sign_label


def _detection_stats(detections: list[dict[str, Any]], elapsed: float) -> dict[str, float | int]:
    confidences = [float(item.get("confidence", 0.0)) for item in detections or []]
    return {
        "inference_time_ms": elapsed * 1000,
        "fps": 1 / elapsed if elapsed > 0 else 0,
        "detection_count": len(detections or []),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
    }


def _render_detection_metrics(stats: dict[str, float | int], prefix: str = "") -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}Inference time", f"{float(stats['inference_time_ms']):.1f} ms")
    c2.metric(f"{prefix}FPS ước tính", f"{float(stats['fps']):.2f}")
    c3.metric(f"{prefix}Số detection", int(stats["detection_count"]))
    c4.metric(f"{prefix}Confidence TB", f"{float(stats['avg_confidence']):.3f}")


def _format_file_size(path: Path) -> str:
    if not path.exists():
        return "Không đọc được"
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def render_image_tab(
    model,
    signs_data: dict,
    conf_threshold: float,
    img_size: int,
    enable_speech: bool,
    save_history: bool,
    show_performance: bool,
    selected_vehicle_type: str,
) -> None:
    """Render image upload and detection workflow."""
    st.session_state.setdefault("image_results", None)
    st.session_state.setdefault("image_file_name", None)
    st.session_state.setdefault("image_is_processing", False)
    st.session_state.setdefault("image_pending_detect", False)
    st.session_state.setdefault("image_last_error", None)
    st.session_state.setdefault("image_last_status", "idle")

    st.markdown("<div class='section-title'>Nhận diện từ ảnh</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Tải ảnh để AI nhận diện biển báo và xem giải thích học tập.</div>",
        unsafe_allow_html=True,
    )

    stepper_placeholder = st.empty()

    with st.container(border=True):
        st.markdown("<div class=\"panel-title\">Chọn ảnh biển báo</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"panel-subtitle\">Kéo thả hoặc chọn file ảnh để bắt đầu.</div>",
            unsafe_allow_html=True,
        )
        uploaded_image = st.file_uploader(
            "Chọn ảnh biển báo",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        has_image = uploaded_image is not None or bool(st.session_state.image_file_name)
        status_state = st.session_state.image_last_status
        if st.session_state.image_is_processing:
            status_state = "processing"

        step_one_class = "step--done" if has_image else ""
        step_two_class = "step--active" if status_state == "processing" else ("step--done" if status_state == "done" else "")
        step_three_class = ""
        if status_state == "done":
            step_three_class = "step--done"
        elif status_state == "no_detection":
            step_three_class = "step--warning"
        elif status_state == "error":
            step_three_class = "step--error"

        stepper_placeholder.markdown(
            f"""
            <div class="stepper">
                <div class="step {step_one_class}"><strong>1.</strong> Tải ảnh</div>
                <div class="step {step_two_class}"><strong>2.</strong> AI nhận diện</div>
                <div class="step {step_three_class}"><strong>3.</strong> Xem giải thích</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        button_label = "Đang nhận diện..." if st.session_state.image_is_processing else (
            "Nhận diện lại" if st.session_state.image_results else "Chạy nhận diện"
        )
        run_detect = st.button(
            button_label,
            type="primary",
            disabled=st.session_state.image_is_processing or uploaded_image is None,
        )

    if not uploaded_image:
        st.session_state.image_last_status = "idle"
        st.session_state.image_is_processing = False
        st.session_state.image_pending_detect = False
        st.markdown(
            "<div class=\"empty-state\"><strong>Chưa có kết quả nhận diện</strong><br/>Hãy tải ảnh biển báo và bấm Chạy AI nhận diện để bắt đầu.</div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.image_file_name != uploaded_image.name:
        st.session_state.image_file_name = uploaded_image.name
        st.session_state.image_results = None
        st.session_state.image_last_error = None
        st.session_state.image_last_status = "selected"

    if run_detect and not st.session_state.image_is_processing:
        st.session_state.image_pending_detect = True
        st.session_state.image_is_processing = True
        st.rerun()

    try:
        image = Image.open(uploaded_image).convert("RGB")
        image_rgb = np.array(image)
        if model is None:
            st.session_state.image_is_processing = False
            st.session_state.image_pending_detect = False
            st.info("Chưa thể nhận diện vì model chưa sẵn sàng.")
            return

        if st.session_state.image_pending_detect:
            st.session_state.image_pending_detect = False
            st.session_state.image_last_error = None
            st.session_state.image_last_status = "processing"
            with st.spinner("Đang nhận diện biển báo..."):
                start = time.time()
                detections, annotated = detect_and_render_image(
                    model,
                    image_rgb,
                    signs_data,
                    conf_threshold,
                    img_size,
                )
                elapsed = time.time() - start
            st.session_state.image_results = {
                "image": image_rgb,
                "annotated": annotated,
                "detections": detections,
                "elapsed": elapsed,
            }
            st.session_state.image_is_processing = False
            st.session_state.image_last_status = "done" if detections else "no_detection"
            if detections:
                st.toast(f"Đã nhận diện {len(detections)} biển báo trong ảnh.", icon="✅")
            else:
                st.toast(
                    "Chưa phát hiện biển báo phù hợp. Hãy thử ảnh rõ hơn hoặc giảm ngưỡng tin cậy.",
                    icon="⚠️",
                )

        results = st.session_state.image_results
        if not results:
            st.markdown(
                "<div class=\"empty-state\">Tải ảnh và bấm “Chạy nhận diện” để xem kết quả.</div>",
                unsafe_allow_html=True,
            )
            return

        detections = results["detections"]
        summary_card = ""
        if detections:
            best_conf = max(float(item.get("confidence", 0.0)) for item in detections)
            summary_card = (
                f"AI phát hiện {len(detections)} biển báo · "
                f"Cao nhất: {best_conf:.2f} · "
                f"Phương tiện: {get_vehicle_label(selected_vehicle_type)} · "
                f"Ngưỡng: {conf_threshold:.2f}"
            )
        else:
            summary_card = (
                f"Chưa phát hiện biển báo · "
                f"Phương tiện: {get_vehicle_label(selected_vehicle_type)} · "
                f"Ngưỡng: {conf_threshold:.2f}"
            )

        st.markdown(
            f"<div class='section-card result-highlight'><div class='panel-title'>Tóm tắt nhanh</div>{summary_card}</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a.container(border=True):
            st.markdown("<div class=\"panel-title\">Ảnh gốc</div>", unsafe_allow_html=True)
            st.image(results["image"], use_container_width=True)
        with col_b.container(border=True):
            st.markdown("<div class=\"panel-title\">Kết quả AI</div>", unsafe_allow_html=True)
            st.image(results["annotated"], use_container_width=True)

        if show_performance:
            with st.expander("Hiệu năng inference", expanded=False):
                _render_detection_metrics(_detection_stats(detections, results["elapsed"]))

        if not detections:
            st.markdown(
                "<div class=\"empty-state\">Không phát hiện biển báo trong ảnh.</div>",
                unsafe_allow_html=True,
            )
            return

        for idx, detection in enumerate(detections):
            sign_info = get_sign_info(detection["class_id"], signs_data)
            st.session_state.last_sign_info = sign_info
            save_detection_with_cooldown(detection, sign_info, "image", seconds=1, enabled=save_history)

        primary_detection = max(detections, key=lambda item: float(item.get("confidence", 0.0)))
        primary_sign = get_sign_info(primary_detection["class_id"], signs_data)
        st.session_state.last_sign_info = primary_sign
        st.session_state.last_detections = detections

        with st.container(border=True):
            st.markdown("<div class='panel-title'>Tóm tắt kết quả</div>", unsafe_allow_html=True)
            rows = []
            for detection in detections:
                sign_info = get_sign_info(detection["class_id"], signs_data)
                short_name = sign_info.get("short_name") or detection.get("class_name", "Biển báo")
                rows.append(
                    {
                        "Biển báo": short_name,
                        "Độ tin cậy": f"{float(detection.get('confidence', 0.0)):.2f}",
                        "Ý nghĩa": sign_info.get("meaning", ""),
                        "Cảnh báo": generate_warning(sign_info),
                    }
                )
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        option_labels = []
        for item in detections:
            sign_info = get_sign_info(item["class_id"], signs_data)
            label_name = sign_info.get("short_name") or item.get("class_name", "Biển báo")
            option_labels.append(f"{label_name} · {float(item.get('confidence', 0.0)):.2f}")
        selected_label = st.selectbox(
            "Chọn biển báo để xem chi tiết",
            options=option_labels,
        )
        selected_index = option_labels.index(selected_label) if selected_label in option_labels else 0
        selected_detection = detections[selected_index]
        selected_sign = get_sign_info(selected_detection["class_id"], signs_data)
        info_col, penalty_col = st.columns(2)
        with info_col:
            render_explanation_card(selected_sign)
        with penalty_col:
            render_penalty_card(
                selected_sign,
                selected_vehicle_type,
                key_prefix=f"img_selected_{selected_sign.get('class_id')}",
            )

        action_col_1, action_col_2, action_col_3, action_col_4 = st.columns(4)
        action_col_1.button("Xem giải thích", disabled=False)
        action_col_2.button("Hỏi chatbot", disabled=True, help="Sẽ bổ sung sau")
        action_col_3.button("Tạo quiz", disabled=True, help="Sẽ bổ sung sau")
        action_col_4.button("Lưu vào lịch sử", disabled=True, help="Sẽ bổ sung sau")

    except Exception as exc:
        st.session_state.image_last_error = "Không xử lý được ảnh đã tải. Vui lòng thử ảnh khác hoặc kiểm tra định dạng."
        st.session_state.image_last_status = "error"
        st.session_state.image_is_processing = False
        st.session_state.image_pending_detect = False
        st.error(st.session_state.image_last_error)


def render_video_tab(
    model,
    signs_data: dict,
    conf_threshold: float,
    img_size: int,
    enable_speech: bool,
    save_history: bool,
    show_performance: bool,
    video_stride: int,
    selected_vehicle_type: str,
) -> None:
    """Render video upload and frame-by-frame detection workflow."""
    st.header("Nhận diện từ video")
    st.caption("Xử lý video theo từng frame mẫu để demo luồng phát hiện và lưu lịch sử nhận diện.")
    uploaded_video = st.file_uploader("Chọn video giao thông", type=["mp4", "avi", "mov", "mkv", "webm"])
    max_frames = st.number_input("Giới hạn frame xử lý", min_value=30, max_value=3000, value=600, step=30)

    if not uploaded_video:
        return
    if model is None:
        st.info("Chưa thể detect video vì model chưa sẵn sàng.")
        return
    if not st.button("Nhận diện video", type="primary"):
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
    processed_frames = 0
    total_detections = 0
    inference_times: list[float] = []
    started_at = time.time()
    last_seen: dict[int, dict] = {}
    last_audio_path = None

    while frame_index < int(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_index % int(video_stride) == 0:
            try:
                inference_start = time.time()
                detections = detect_image(model, frame_bgr, conf_threshold, img_size)
                apply_short_labels(detections, signs_data)
                inference_elapsed = time.time() - inference_start
                processed_frames += 1
                total_detections += len(detections)
                inference_times.append(inference_elapsed)
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
                    save_detection_with_cooldown(
                        detection,
                        sign_info,
                        "video",
                        seconds=8,
                        enabled=save_history,
                    )
                    if enable_speech and should_speak(detection["class_id"], cooldown_seconds=8):
                        last_audio_path = text_to_speech(generate_speech_text(sign_info), AUDIO_DIR)
            except Exception as exc:
                st.warning(f"Bỏ qua frame {frame_index} do lỗi inference: {exc}")

        frame_index += 1
        progress.progress(min(frame_index / int(max_frames), 1.0))

    cap.release()
    progress.empty()
    total_elapsed = time.time() - started_at
    if show_performance:
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_fps = processed_frames / sum(inference_times) if inference_times and sum(inference_times) > 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FPS TB inference", f"{avg_fps:.2f}")
        c2.metric("Frame đã detect", processed_frames)
        c3.metric("Tổng detection", total_detections)
        c4.metric("Thời gian xử lý", f"{total_elapsed:.1f}s")
        st.caption(f"Inference trung bình: {avg_inference * 1000:.1f} ms/frame detect")

    if not last_seen:
        st.warning("Không phát hiện biển báo trong video hoặc trong số frame đã xử lý.")
        return

    best_item = max(
        last_seen.values(),
        key=lambda item: float(item.get("detection", {}).get("confidence", 0.0)),
    )
    st.session_state.last_sign_info = best_item.get("sign_info")
    st.session_state.last_detections = [item.get("detection") for item in last_seen.values()]

    with result_placeholder:
        st.subheader("Biển báo đã phát hiện")
        for item in last_seen.values():
            with st.expander(item["sign_info"].get("class_name", "Biển báo")):
                show_sign_info(
                    item["sign_info"],
                    item["detection"],
                    selected_vehicle_type,
                    key_prefix=f"video_{item['detection'].get('class_id')}",
                )
        if enable_speech and last_audio_path:
            st.audio(last_audio_path)


def _find_metric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        column: str(column).lower().replace(" ", "").replace("_", "").replace("-", "")
        for column in df.columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "").replace("-", "")
        for column, normalized_name in normalized.items():
            if key in normalized_name:
                return str(column)
    return None


def render_model_evaluation_tab(model, classes: list[str]) -> None:
    """Render model metadata and YOLO training/validation artifacts."""
    st.header("Đánh giá model")
    st.caption("Tổng hợp nhanh trạng thái model và các file đánh giá sinh ra từ YOLO train/val nếu có.")

    model_exists = MODEL_PATH.exists()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model path", str(MODEL_PATH.relative_to(BASE_DIR)))
    c2.metric("Số class", len(classes) if classes else 0)
    c3.metric("Framework", "YOLO")
    c4.metric("Trạng thái", "loaded" if model is not None else ("missing" if not model_exists else "error"))

    c5, c6 = st.columns(2)
    c5.metric("Dataset", "Vietnamese Traffic Signs")
    c6.metric("Kích thước file model", _format_file_size(MODEL_PATH))
    if not model_exists:
        st.warning("Chưa tìm thấy models/best.pt. Hãy đặt model đúng đường dẫn để chạy nhận diện.")

    artifact_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.csv",
    ]
    candidate_dirs = [
        BASE_DIR / "runs" / "detect" / "train",
        BASE_DIR / "runs" / "detect" / "train2",
        BASE_DIR / "runs" / "detect" / "val",
        BASE_DIR / "runs" / "train",
    ]
    found_any = False

    st.subheader("File kết quả train/val")
    for directory in candidate_dirs:
        existing = [directory / name for name in artifact_names if (directory / name).exists()]
        if not existing:
            continue
        found_any = True
        with st.expander(str(directory.relative_to(BASE_DIR)), expanded=True):
            image_files = [path for path in existing if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            for row_start in range(0, len(image_files), 2):
                cols = st.columns(2)
                for col, image_path in zip(cols, image_files[row_start : row_start + 2]):
                    col.image(str(image_path), caption=image_path.name, use_container_width=True)

            csv_path = directory / "results.csv"
            if csv_path.exists():
                try:
                    metrics_df = pd.read_csv(csv_path)
                    st.markdown("#### Bảng metrics")
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                    if not metrics_df.empty:
                        last_row = metrics_df.iloc[-1]
                        metric_columns = {
                            "Precision": _find_metric_column(metrics_df, ["metrics/precision", "precision"]),
                            "Recall": _find_metric_column(metrics_df, ["metrics/recall", "recall"]),
                            "mAP50": _find_metric_column(metrics_df, ["metrics/map50", "map50"]),
                            "mAP50-95": _find_metric_column(metrics_df, ["metrics/map50-95", "map5095"]),
                        }
                        cols = st.columns(4)
                        for col, (label, column) in zip(cols, metric_columns.items()):
                            value = last_row[column] if column and column in metrics_df else None
                            col.metric(label, f"{float(value):.4f}" if pd.notna(value) else "N/A")

                        chart_cols = [column for column in metric_columns.values() if column in metrics_df]
                        if chart_cols:
                            st.line_chart(metrics_df[chart_cols])
                except Exception as exc:
                    st.warning(f"Không đọc được results.csv trong {directory.name}: {exc}")

    if not found_any:
        st.info("Chưa tìm thấy file đánh giá model. Bạn có thể chạy val hoặc train để sinh các biểu đồ này.")


def render_webcam_tab(
    model,
    signs_data: dict,
    conf_threshold: float,
    img_size: int,
    enable_speech: bool,
    save_history: bool,
    show_performance: bool,
    video_stride: int,
    selected_vehicle_type: str,
) -> None:
    """Render a Streamlit WebRTC webcam demo with safe dependency fallback."""
    st.header("Webcam realtime")
    st.caption("Demo detect trực tiếp bằng webcam. Tính năng này chạy tốt nhất khi mở app trên máy local có camera.")

    if model is None:
        st.info("Chưa thể chạy webcam vì model chưa sẵn sàng.")
        return

    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except Exception:
        st.warning("Chưa cài dependency webcam. Hãy cài `streamlit-webrtc` và `av`, rồi chạy lại app.")
        st.code("pip install streamlit-webrtc av", language="bash")
        return

    class TrafficSignVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.frame_index = 0
            self.processed_frames = 0
            self.total_detections = 0
            self.total_inference_time = 0.0
            self.last_inference_time = 0.0
            self.latest_detections: list[dict[str, Any]] = []
            self.latest_signs: list[dict[str, Any]] = []
            self.history_cooldown: dict[int, float] = {}

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            frame_bgr = frame.to_ndarray(format="bgr24")
            annotated = frame_bgr
            should_process = self.frame_index % max(int(video_stride), 1) == 0

            if should_process:
                try:
                    started_at = time.time()
                    detections = detect_image(model, frame_bgr, conf_threshold, img_size)
                    elapsed = time.time() - started_at
                    annotated = draw_detections(frame_bgr, detections, signs_data)
                    sign_infos = [get_sign_info(item["class_id"], signs_data) for item in detections]

                    now = time.time()
                    if save_history:
                        for detection, sign_info in zip(detections, sign_infos):
                            class_id = int(detection.get("class_id", -1))
                            last_saved = self.history_cooldown.get(class_id, 0)
                            if now - last_saved >= 8:
                                save_detection(
                                    class_id,
                                    sign_info.get("class_name", detection.get("class_name", "Unknown")),
                                    float(detection.get("confidence", 0.0)),
                                    sign_info.get("meaning", ""),
                                    "webcam",
                                    DB_PATH,
                                )
                                self.history_cooldown[class_id] = now

                    with self.lock:
                        self.processed_frames += 1
                        self.total_detections += len(detections)
                        self.total_inference_time += elapsed
                        self.last_inference_time = elapsed
                        self.latest_detections = detections
                        self.latest_signs = sign_infos
                except Exception:
                    annotated = frame_bgr
            else:
                with self.lock:
                    if self.latest_detections:
                        annotated = draw_detections(frame_bgr, self.latest_detections, signs_data)

            with self.lock:
                fps = 1 / self.last_inference_time if self.last_inference_time > 0 else 0
                detection_count = len(self.latest_detections)
            if show_performance:
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} | Detect: {detection_count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (30, 220, 120),
                    2,
                    cv2.LINE_AA,
                )

            self.frame_index += 1
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        def get_stats(self) -> dict[str, Any]:
            with self.lock:
                avg_fps = (
                    self.processed_frames / self.total_inference_time
                    if self.total_inference_time > 0
                    else 0
                )
                return {
                    "processed_frames": self.processed_frames,
                    "total_detections": self.total_detections,
                    "avg_fps": avg_fps,
                    "last_inference_time": self.last_inference_time,
                    "latest_detections": list(self.latest_detections),
                    "latest_signs": list(self.latest_signs),
                }

    ctx = webrtc_streamer(
        key="traffic-sign-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=TrafficSignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not ctx.state.playing:
        st.info("Bấm START để bật webcam. Trình duyệt có thể hỏi quyền truy cập camera.")
        return

    processor = ctx.video_processor
    if not processor:
        st.info("Đang khởi tạo webcam...")
        return

    stats = processor.get_stats()
    if show_performance:
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS realtime TB", f"{stats['avg_fps']:.2f}")
        c2.metric("Frame đã detect", stats["processed_frames"])
        c3.metric("Tổng detection", stats["total_detections"])
        st.caption(f"Inference gần nhất: {stats['last_inference_time'] * 1000:.1f} ms")

    latest_signs = stats.get("latest_signs") or []
    latest_detections = stats.get("latest_detections") or []
    if latest_signs:
        st.subheader("Cảnh báo gần nhất")
        for idx, (sign_info, detection) in enumerate(zip(latest_signs[:3], latest_detections[:3])):
            show_sign_info(
                sign_info,
                detection,
                selected_vehicle_type,
                key_prefix=f"webcam_{idx}_{detection.get('class_id')}",
            )
        if enable_speech and st.button("Đọc cảnh báo webcam gần nhất"):
            class_id = latest_signs[0].get("class_id")
            if should_speak(class_id, cooldown_seconds=8):
                speak_sign(latest_signs[0])
            else:
                st.info("Cảnh báo này vừa được đọc, vui lòng chờ cooldown.")


def render_lookup_tab(signs_data: dict, enable_speech: bool, selected_vehicle_type: str) -> None:
    """Render sign lookup workflow."""
    st.header("Tra cứu biển báo")
    st.caption("Chọn một biển trong knowledge base để học ý nghĩa, hành động cần làm và mức phạt tham khảo.")
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
    show_sign_info(
        selected_info,
        vehicle_type=selected_vehicle_type,
        key_prefix=f"lookup_{selected_info.get('class_id')}",
    )
    if enable_speech and st.button("Đọc cảnh báo tra cứu"):
        speak_sign(selected_info)


def render_chat_tab(signs_data: dict, selected_vehicle_type: str) -> None:
    """Render rule-based chatbot workflow."""
    st.markdown("<div class='section-title'>Chatbot hỏi đáp</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Chọn biển báo để AI giải thích ý nghĩa, hành động cần làm, mức phạt tham khảo hoặc tạo ví dụ tình huống.</div>",
        unsafe_allow_html=True,
    )

    if not signs_data:
        st.error("Không tải được data/signs.json.")
        return

    st.session_state.setdefault("selected_sign_code", None)
    st.session_state.setdefault("selected_sign_name", None)
    st.session_state.setdefault("selected_sign_group", None)
    st.session_state.setdefault("current_learning_sign", None)
    st.session_state.setdefault("chat_vehicle_type", selected_vehicle_type)

    def _group_for_sign(info: dict) -> str:
        group_text = str(info.get("type", "")).lower()
        if "cấm" in group_text:
            return "Biển cấm"
        if "nguy hiểm" in group_text or "cảnh báo" in group_text:
            return "Biển nguy hiểm"
        if "hiệu lệnh" in group_text:
            return "Biển hiệu lệnh"
        if "chỉ dẫn" in group_text:
            return "Biển chỉ dẫn"
        if "phụ" in group_text:
            return "Biển phụ"
        return "Khác"

    def _sign_label(entry: tuple[str, dict]) -> str:
        _, info = entry
        code = info.get("code") or f"ID {info.get('class_id', '')}"
        name = info.get("class_name") or info.get("short_name", "Biển báo")
        return f"{code} - {name}"

    last_detected = st.session_state.get("last_sign_info")
    if last_detected and not st.session_state.selected_sign_code:
        st.session_state.selected_sign_code = str(last_detected.get("class_id"))

    group_options = [
        "Tất cả",
        "Biển cấm",
        "Biển nguy hiểm",
        "Biển hiệu lệnh",
        "Biển chỉ dẫn",
        "Biển phụ",
        "Khác",
    ]
    group_col, sign_col, vehicle_col = st.columns([1, 2.2, 1])
    with group_col:
        selected_group = st.selectbox("Nhóm biển báo", options=group_options, key="selected_sign_group")

    sign_items = sorted(signs_data.items(), key=lambda item: int(item[0]))
    if selected_group != "Tất cả":
        sign_items = [item for item in sign_items if _group_for_sign(item[1]) == selected_group]

    if not sign_items:
        st.warning("Không có biển báo phù hợp với nhóm đã chọn.")
        return

    selected_index = 0
    if st.session_state.selected_sign_code:
        for idx, (class_id, _) in enumerate(sign_items):
            if str(class_id) == str(st.session_state.selected_sign_code):
                selected_index = idx
                break

    with sign_col:
        sign_choice = st.selectbox(
            "Chọn biển báo",
            options=sign_items,
            format_func=_sign_label,
            index=selected_index,
            key="chat_sign_select",
        )

    with vehicle_col:
        vehicle_options = ["car", "motorbike", "truck", "bus", "all"]
        vehicle_index = vehicle_options.index(selected_vehicle_type) if selected_vehicle_type in vehicle_options else 0
        selected_vehicle = st.selectbox(
            "Phương tiện",
            options=vehicle_options,
            format_func=get_vehicle_label,
            index=vehicle_index,
            key="chat_vehicle_type",
        )

    selected_info = sign_choice[1]
    st.session_state.current_learning_sign = selected_info
    st.session_state.selected_sign_code = str(selected_info.get("class_id"))
    st.session_state.selected_sign_name = selected_info.get("class_name", "Biển báo")

    has_penalty = bool(selected_info.get("penalty_refs")) or selected_info.get("speed_limit_value") is not None
    penalty_note = "Có mức phạt tham khảo" if has_penalty else "Chưa có mức phạt cụ thể"
    st.markdown(
        f"""
        <div class='section-card'>
            <div class='panel-title'>Ngữ cảnh câu hỏi</div>
            <div><strong>Mã biển:</strong> {selected_info.get('code', '')}</div>
            <div><strong>Tên biển:</strong> {selected_info.get('class_name', 'Biển báo')}</div>
            <div><strong>Nhóm biển:</strong> {_group_for_sign(selected_info)}</div>
            <div><strong>Ý nghĩa ngắn:</strong> {selected_info.get('meaning', 'Chưa có dữ liệu ý nghĩa.')}</div>
            <div><strong>Trạng thái mức phạt:</strong> {penalty_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Đang hỏi về: {selected_info.get('code', '')} - {selected_info.get('class_name', 'Biển báo')} | "
        f"Phương tiện: {get_vehicle_label(selected_vehicle)}"
    )

    code_label = selected_info.get("code", "")
    name_label = selected_info.get("class_name", "Biển báo")
    quick_prompts = [
        ("Biển này có ý nghĩa gì?", f"Hãy giải thích biển {code_label} - {name_label} cho người mới học lái xe."),
        ("Gặp biển này phải làm gì?", f"Gặp biển {code_label} - {name_label} thì cần làm gì?"),
        ("Có bị phạt không?", f"Biển {code_label} - {name_label} có mức phạt tham khảo không?"),
        ("Cho ví dụ tình huống", f"Cho một ví dụ tình huống khi gặp biển {code_label} - {name_label}."),
        ("Tạo câu hỏi quiz từ biển này", f"Tạo một câu hỏi quiz ngắn về biển {code_label} - {name_label}."),
    ]
    quick_cols = st.columns(5)
    quick_question = None
    for col, (label, prompt) in zip(quick_cols, quick_prompts):
        if col.button(label, key=f"chat_quick_{label}"):
            quick_question = prompt

    question = st.chat_input(
        "Hỏi về ý nghĩa, hành động cần làm, mức phạt, căn cứ pháp lý hoặc ví dụ tình huống..."
    )
    asked_question = question or quick_question
    if asked_question:
        actual_speed = st.session_state.last_speed_values.get(str(selected_info.get("class_id")))
        answer = answer_question(
            asked_question,
            selected_info,
            vehicle_type=selected_vehicle,
            actual_speed=actual_speed,
        )
        st.session_state.chat_history.append(("user", asked_question))
        st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history[-12:]:
        with st.chat_message(role):
            st.markdown(message)


def render_quiz_tab(signs_data: dict, scenarios: dict) -> None:
    """Render quiz and learning scenario workflow."""
    st.header("Quiz / Tình huống học tập")
    st.caption("Luyện tập nhanh với các tình huống thường gặp trong bộ dữ liệu biển báo.")
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
    with st.container(border=True):
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
    st.caption("Theo dõi các lượt nhận diện đã lưu trong SQLite và xuất dữ liệu demo khi cần.")
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
