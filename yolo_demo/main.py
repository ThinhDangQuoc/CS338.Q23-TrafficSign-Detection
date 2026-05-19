import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import base64
import uuid
import shutil

app = FastAPI()

# Setup base directory (yolo_demo directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Setup project directory (CS338.Q23-TrafficSign-Detection)
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Path to the YOLOv11 model
MODEL_PATH = os.path.join(PROJECT_DIR, "trained_weights", "best.pt")

# Initialize model variable
model = None

# Setup directories
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def load_model():
    global model
    model = YOLO(MODEL_PATH)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Form(0.25)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
    # Save uploaded file
    file_extension = file.filename.split(".")[-1]
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run inference
    results = model.predict(file_path, conf=conf)
    
    # Process results
    res = results[0]
    img = res.plot() # Image with bounding boxes
    
    # Convert image to base64 to send to frontend
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Get detection info
    detections = []
    for box in res.boxes:
        detections.append({
            "class_name": res.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": [float(x) for x in box.xyxy[0].tolist()]
        })
    
    # Clean up uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return JSONResponse({
        "image": img_base64,
        "detections": detections,
        "count": len(detections)
    })

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
