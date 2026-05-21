"""YOLO detection helpers for images and video frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_sign_app.config import CACHE_DIR, CLASSES_PATH
from traffic_sign_app.services.knowledge_base import load_classes


def load_model(model_path: str = "models/best.pt"):
    """Load a YOLO model from disk with a clear error when it is missing."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {path}. Hãy đặt file best.pt vào thư mục models/."
        )

    CACHE_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Chưa cài ultralytics. Hãy chạy: pip install -r requirements.txt") from exc

    return YOLO(str(path))


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def detect_image(model, image, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
    """Run YOLO on a BGR/RGB image array and return normalized detections."""
    if model is None or image is None:
        return []

    frame = np.asarray(image)
    if frame.size == 0:
        return []

    try:
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Lỗi khi chạy YOLO inference: {exc}") from exc

    if not results:
        return []

    class_names = load_classes(CLASSES_PATH)
    names = _model_names(model)
    detections: list[dict[str, Any]] = []

    for box in getattr(results[0], "boxes", []) or []:
        class_id = int(box.cls[0])
        xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else names.get(class_id, f"Class {class_id}")
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "box": xyxy,
            }
        )

    return detections


def draw_detections(image, detections: list[dict[str, Any]], signs_data: dict | None = None):
    """Draw bounding boxes and labels on an image."""
    if image is None:
        return image

    annotated = np.asarray(image).copy()
    height, width = annotated.shape[:2]

    for detection in detections or []:
        x1, y1, x2, y2 = detection.get("box", [0, 0, 0, 0])
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        class_id = detection.get("class_id")
        sign_info = (signs_data or {}).get(str(class_id), {})
        label_name = sign_info.get("short_name") or detection.get("class_name", f"Class {class_id}")
        confidence = float(detection.get("confidence", 0.0))
        label = f"{class_id}: {label_name} {confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 160, 90), 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y1 - text_h - 8)
        cv2.rectangle(annotated, (x1, label_y), (min(width - 1, x1 + text_w + 8), y1), (20, 160, 90), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_h + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated

