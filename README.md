# Traffic Sign Assistant

Hệ thống nhận dạng và cảnh báo biển báo giao thông Việt Nam sử dụng YOLOv11s.

Ứng dụng mở rộng demo YOLO ban đầu thành một web app FastAPI có upload ảnh/video, vẽ bounding box, sinh cảnh báo tiếng Việt, lưu lịch sử vào SQLite, lưu snapshot và xuất báo cáo CSV.

## Chức năng

- Upload ảnh để nhận dạng biển báo.
- Upload video và xuất video đã vẽ bounding box.
- Hiển thị tên biển báo, confidence và cảnh báo tiếng Việt.
- Lưu snapshot từng biển báo phát hiện vào `yolo_demo/static/snapshots/`.
- Lưu ảnh/video kết quả vào `yolo_demo/static/outputs/`.
- Lưu lịch sử nhận dạng bằng SQLite tại `yolo_demo/detections.db`.
- Dashboard thống kê tổng số phát hiện, số lượng theo class và lịch sử gần đây.
- Xuất báo cáo CSV tại endpoint `/report`.

## Cấu trúc chính

```text
CS338.Q23-TrafficSign-Detection/
├── trained_weights/
│   ├── best.pt
│   └── dataset.yaml
├── yolo_demo/
│   ├── main.py
│   ├── detector.py
│   ├── warning_engine.py
│   ├── database.py
│   ├── report.py
│   ├── static/
│   │   ├── index.html
│   │   ├── outputs/
│   │   └── snapshots/
│   └── uploads/
├── requirements.txt
└── vietnam-traffic-sign-with-yolo11s-map-50-0-98.ipynb
```

## Cài đặt

```bash
cd CS338.Q23-TrafficSign-Detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

File `requirements.txt` ưu tiên PyTorch bản CPU để tránh tải các gói CUDA quá nặng khi chỉ cần chạy demo local.

## Chạy ứng dụng

```bash
cd CS338.Q23-TrafficSign-Detection
python yolo_demo/main.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8001
```

Nếu port `8001` đang bận:

```bash
PORT=8765 python yolo_demo/main.py
```

## API chính

- `GET /`: giao diện dashboard.
- `POST /predict`: upload ảnh/video, tham số form `file` và `conf`.
- `GET /dashboard`: lấy lịch sử và thống kê.
- `GET /report`: xuất file `traffic_sign_report.csv`.

## Ghi chú

Model được load từ:

```text
trained_weights/best.pt
```

Tên class được lấy trực tiếp từ model khi inference. File `warning_engine.py` chuẩn hóa tên class để sinh cảnh báo tiếng Việt tương ứng.
