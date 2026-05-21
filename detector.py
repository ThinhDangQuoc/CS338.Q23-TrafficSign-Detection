"""Backward-compatible imports for detector helpers."""

from traffic_sign_app.core.detector import detect_image, draw_detections, load_model

__all__ = ["load_model", "detect_image", "draw_detections"]

