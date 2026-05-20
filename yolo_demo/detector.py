"""YOLO detector wrapper for image and video inference."""

from __future__ import annotations

import os
import uuid
from typing import Any

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "trained_weights", "best.pt")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".matplotlib"))

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class TrafficSignDetector:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find YOLO model at {model_path}")
        self.model = YOLO(model_path)

    def detect_image(
        self,
        input_path: str,
        output_path: str,
        snapshot_dir: str,
        conf: float = 0.5,
    ) -> list[dict[str, Any]]:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Uploaded image cannot be read by OpenCV")

        result = self.model.predict(source=image, conf=conf, verbose=False)[0]
        annotated = result.plot()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        detections = self._parse_detections(result)
        for detection in detections:
            detection["snapshot_path"] = self._save_snapshot(
                image,
                detection["bbox"],
                snapshot_dir,
                detection["class_name"],
            )
        return detections

    def detect_video(
        self,
        input_path: str,
        output_path: str,
        snapshot_dir: str,
        conf: float = 0.5,
        snapshot_interval_seconds: float = 2.0,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Uploaded video cannot be opened by OpenCV")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        if fps <= 1:
            fps = 24.0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        detections: list[dict[str, Any]] = []
        frame_index = 0
        last_snapshot_at: dict[str, float] = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = self.model.predict(source=frame, conf=conf, verbose=False)[0]
            annotated = result.plot()
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)

            timestamp = frame_index / fps
            for detection in self._parse_detections(result):
                sign_name = detection["class_name"]
                previous = last_snapshot_at.get(sign_name, -snapshot_interval_seconds)
                if timestamp - previous >= snapshot_interval_seconds:
                    detection["video_time"] = round(timestamp, 2)
                    detection["snapshot_path"] = self._save_snapshot(
                        frame,
                        detection["bbox"],
                        snapshot_dir,
                        sign_name,
                        frame_index,
                    )
                    detections.append(detection)
                    last_snapshot_at[sign_name] = timestamp

            frame_index += 1

        cap.release()
        writer.release()
        return detections

    def _parse_detections(self, result: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            bbox = [int(round(value)) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf[0]) * 100, 2),
                    "bbox": bbox,
                }
            )
        return detections

    def _save_snapshot(
        self,
        frame: Any,
        bbox: list[int],
        snapshot_dir: str,
        sign_name: str,
        frame_index: int | None = None,
    ) -> str | None:
        os.makedirs(snapshot_dir, exist_ok=True)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        safe_name = "".join(char if char.isalnum() else "_" for char in sign_name.lower())[:40]
        frame_part = f"_f{frame_index}" if frame_index is not None else ""
        filename = f"{safe_name}{frame_part}_{uuid.uuid4().hex[:10]}.jpg"
        path = os.path.join(snapshot_dir, filename)
        cv2.imwrite(path, crop)
        return f"/static/snapshots/{filename}"


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def is_video_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTENSIONS
