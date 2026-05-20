from __future__ import annotations

from contextlib import asynccontextmanager
import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from database import add_detection, get_detection_rows, get_stats, init_db
from detector import TrafficSignDetector, is_image_file, is_video_file
from report import export_csv
from warning_engine import get_warning


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
SNAPSHOT_DIR = os.path.join(STATIC_DIR, "snapshots")

detector: TrafficSignDetector | None = None


def ensure_directories() -> None:
    for directory in (UPLOAD_DIR, OUTPUT_DIR, SNAPSHOT_DIR):
        os.makedirs(directory, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    ensure_directories()
    init_db()
    detector = TrafficSignDetector()
    yield


app = FastAPI(title="Traffic Sign Assistant", lifespan=lifespan)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/dashboard")
async def dashboard() -> JSONResponse:
    return JSONResponse(
        {
            "history": get_detection_rows(limit=40),
            "stats": get_stats(),
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Form(0.5)) -> JSONResponse:
    if detector is None:
        raise HTTPException(status_code=503, detail="YOLO model is still loading")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ensure_directories()
    conf = max(0.05, min(float(conf), 1.0))
    ext = os.path.splitext(file.filename)[1].lower()
    file_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if is_image_file(file.filename):
            source_type = "image"
            output_name = f"{file_id}_detected.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            detections = detector.detect_image(input_path, output_path, SNAPSHOT_DIR, conf=conf)
            output_url = f"/static/outputs/{output_name}"
        elif is_video_file(file.filename):
            source_type = "video"
            output_name = f"{file_id}_detected.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            detections = detector.detect_video(input_path, output_path, SNAPSHOT_DIR, conf=conf)
            output_url = f"/static/outputs/{output_name}"
        else:
            raise HTTPException(status_code=400, detail="Only image and video files are supported")

        for detection in detections:
            warning = get_warning(detection["class_name"])
            detection["warning_message"] = warning
            add_detection(
                sign_name=detection["class_name"],
                confidence=detection["confidence"],
                warning_message=warning,
                snapshot_path=detection.get("snapshot_path"),
                source_type=source_type,
            )

        return JSONResponse(
            {
                "media_type": source_type,
                "output_url": output_url,
                "detections": detections,
                "count": len(detections),
                "history": get_detection_rows(limit=40),
                "stats": get_stats(),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/report")
async def report() -> FileResponse:
    ensure_directories()
    path = export_csv(get_detection_rows(limit=None))
    return FileResponse(path, media_type="text/csv", filename="traffic_sign_report.csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8001")),
    )
