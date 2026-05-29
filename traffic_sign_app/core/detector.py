"""YOLO detection helpers for images and video frames."""

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
import os
from functools import lru_cache
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


@lru_cache(maxsize=4)
def _cached_classes(path: str) -> tuple[str, ...]:
    return tuple(load_classes(path))


def detect_image(
    model,
    image,
    conf_threshold: float = 0.25,
    img_size: int | None = None,
    track: bool = False,
) -> list[dict[str, Any]]:
    """Run YOLO on a BGR/RGB image array and return normalized detections."""
    if model is None or image is None:
        return []

    frame = np.asarray(image)
    if frame.size == 0:
        return []

    try:
        predict_kwargs: dict[str, Any] = {"source": frame, "conf": conf_threshold, "verbose": False}
        if img_size:
            predict_kwargs["imgsz"] = int(img_size)
        if track:
            results = model.track(persist=True, **predict_kwargs)
        else:
            results = model.predict(**predict_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Lỗi khi chạy YOLO inference: {exc}") from exc

    if not results:
        return []

    class_names = list(_cached_classes(str(CLASSES_PATH)))
    names = _model_names(model)
    detections: list[dict[str, Any]] = []

    for box in getattr(results[0], "boxes", []) or []:
        class_id = int(box.cls[0])
        xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else names.get(class_id, f"Class {class_id}")
        
        # Get track id if tracking is enabled and it is assigned
        track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
        
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "box": xyxy,
                "track_id": track_id,
            }
        )

    return detections

def _load_vietnamese_font(size: int = 18):
    """Load a Unicode font that supports Vietnamese."""
    font_candidates = [
        "assets/fonts/DejaVuSans.ttf",
        "assets/fonts/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()
def draw_detections(image, detections: list[dict[str, Any]], signs_data: dict | None = None):
    """Draw bounding boxes and Vietnamese labels on an image."""
    if image is None:
        return image

    annotated = np.asarray(image).copy()
    height, width = annotated.shape[:2]

    # Draw boxes by OpenCV first
    def _shorten_label(text: str, max_len: int = 32) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    for detection in detections or []:
        x1, y1, x2, y2 = detection.get("box", [0, 0, 0, 0])
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 160, 90), 2)

    # Draw Vietnamese text by Pillow
    pil_img = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_img)
    font = _load_vietnamese_font(size=18)

    for detection in detections or []:
        x1, y1, x2, y2 = detection.get("box", [0, 0, 0, 0])
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)

        class_id = detection.get("class_id")
        sign_info = (signs_data or {}).get(str(class_id), {})
        label_name = sign_info.get("short_name") or detection.get("class_name", f"Class {class_id}")
        confidence = float(detection.get("confidence", 0.0))
        clean_name = _shorten_label(str(label_name), max_len=36)
        label = f"{clean_name} · {confidence:.2f}"

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = x1
        label_y = max(0, y1 - text_h - 10)

        bg_x2 = min(width - 1, label_x + text_w + 10)
        bg_y2 = min(height - 1, label_y + text_h + 8)

        draw.rectangle(
            [(label_x, label_y), (bg_x2, bg_y2)],
            fill=(20, 160, 90),
        )

        draw.text(
            (label_x + 5, label_y + 3),
            label,
            font=font,
            fill=(255, 255, 255),
        )

    return np.array(pil_img)