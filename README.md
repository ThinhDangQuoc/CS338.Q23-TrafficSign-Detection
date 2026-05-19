# YOLOv11 Vietnamese Traffic Sign Detection & Dashboard

An interactive, real-time road sign detection system utilizing the state-of-the-art **YOLOv11s** model. The model is trained on a comprehensive Vietnamese traffic sign dataset, featuring advanced data preprocessing, deduplication, and rare-class augmentation strategies to achieve an outstanding **98% mAP@50** accuracy. 

It is integrated with a high-performance **FastAPI** backend and a premium, responsive **glassmorphic real-time web dashboard** for inference visualization.

---

## 🚀 Key Features

* **YOLOv11s Core Architecture**: High-speed, high-accuracy object detection tuned for complex Vietnamese road settings.
* **Outstanding Performance**: Reached a **98% mAP@50** and **50% mAP@50-95** score on the Vietnamese traffic signs validation split.
* **Advanced Data Augmentations**: Built-in pHash-based deduplication to clean up over-represented classes, and Albumentations strong augmentations to handle rare and minority classes (speed limit signs, caution indicators, etc.).
* **Glassmorphic Web Dashboard**: 
  * Simple **Drag & Drop** image uploader (supports PNG, JPG).
  * **Interactive Confidence Slider** to instantly adjust inference sensitivity (0.05 to 1.00).
  * Real-time **base64 bounding box plotting** on images.
  * Sleek detection metrics sidebar presenting detected classes and confidence ratings.
* **Portable & Zero-GPU Overhead**: Runs seamlessly on CPU-only machines with path-independent local configurations.

---

## 📂 Project Structure

```
CS338.Q23-TrafficSign-Detection/
├── .venv/                         # Python virtual environment (Python 3.9)
├── .gitignore                     # Git configuration to ignore envs, uploads, and checkpoints
├── trained_weights/               # Directory containing the model weights and Kaggle configurations
│   ├── best.pt                    # Custom YOLOv11s traffic sign weights (98% mAP)
│   └── dataset.yaml               # Training dataset mapping configurations
├── yolo_demo/                     # FastAPI Real-time Web Dashboard Application
│   ├── main.py                    # Server logic for inference & serving frontend
│   ├── static/                    # Frontend assets
│   │   └── index.html             # Beautiful Glassmorphic Web UI
│   └── uploads/                   # Temporary upload directory (ignored by git, kept by .gitkeep)
└── vietnam-traffic-sign-with-yolo11s-map-50-0-98.ipynb   # Jupyter training notebook
```

---

## 🛠️ Installation & Setup

Set up your workspace locally in under a minute with Python 3.9:

### 1. Clone the repository and navigate to the project directory:
```bash
cd CS338.Q23-TrafficSign-Detection
```

### 2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 3. Install the dependencies:
We utilize the lightweight CPU-only PyTorch version along with `ultralytics` and `fastapi` for speed and ease of local hosting without requiring massive 2.5 GB CUDA packages:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics fastapi uvicorn python-multipart
```

---

## 🖥️ Running the Dashboard

Once the virtual environment is set up and activated, spin up the server locally:

```bash
python yolo_demo/main.py
```

After starting up, open your web browser and navigate to:
👉 **[http://localhost:8001](http://localhost:8001)**

Now, upload any road scene image, set your confidence threshold, and watch the YOLOv11s model detect and classify traffic signs in real time!

---

## 📊 Dataset & Preprocessing

The model has been optimized with strict datasets pipeline:
1. **Deduplication**: Using Perceptual Hashing (`phash`) to eliminate duplicate image frames in dense classification slots.
2. **Rare-Class Augmentation**: Balancing highly under-represented classes by applying extra passes of specialized albumentations (GaussianBlur, Rotate, RandomBrightnessContrast, ShiftScaleRotate).
3. **Training & Validation**: Split using stratified configurations to guarantee validation consistency.
