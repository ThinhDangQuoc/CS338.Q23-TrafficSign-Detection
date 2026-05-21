# Hệ thống học tập nhận diện biển báo giao thông Việt Nam bằng YOLO

Đây là hệ thống hỗ trợ học tập, nhận diện và giải thích biển báo giao thông trong môi trường mô phỏng, ảnh, video hoặc camera. Ứng dụng dùng model YOLO đã train trên bộ dữ liệu Kaggle Vietnamese Traffic Signs để phát hiện biển báo, sau đó tra cứu nội dung trong JSON và hiển thị giải thích bằng tiếng Việt.

Ứng dụng không phải hệ thống lái xe an toàn tuyệt đối và không thay thế thiết bị an toàn thực tế trên xe.

## Mục tiêu

- Nhận diện biển báo giao thông Việt Nam từ ảnh và video.
- Hiển thị bounding box, class ID, tên biển và confidence.
- Giải thích ý nghĩa biển báo, hành động người lái cần làm, cảnh báo ngắn, lỗi thường gặp và mức phạt tham khảo theo loại phương tiện.
- Đọc cảnh báo bằng giọng nói nếu TTS hoạt động.
- Lưu lịch sử nhận diện bằng SQLite.
- Có chatbot rule-based để hỏi đáp theo biển báo.
- Có quiz/tình huống học tập đơn giản.

## Công nghệ sử dụng

- Python
- Ultralytics YOLO
- OpenCV, NumPy, Pillow
- Streamlit
- JSON cho dữ liệu kiến thức biển báo
- SQLite cho lịch sử nhận diện
- edge-tts hoặc gTTS cho speech
- pandas cho bảng lịch sử và export CSV

## Pipeline hoạt động

```text
Image / Video / Webcam
        ↓
OpenCV đọc ảnh hoặc frame video
        ↓
YOLO detect biển báo
        ↓
Lấy class_id, class_name, confidence, bounding_box
        ↓
Vẽ bounding box lên ảnh/frame
        ↓
Tra class_id trong data/signs.json
        ↓
Sinh cảnh báo và giải thích
        ↓
Streamlit hiển thị kết quả
        ↓
TTS đọc cảnh báo nếu bật
        ↓
SQLite lưu lịch sử nhận diện
        ↓
Chatbot và Quiz hỗ trợ học tập
```

## Cấu trúc thư mục

```text
CS338.Q23-TrafficSign-Detection/
├── main.py
├── traffic_sign_app/
│   ├── config.py
│   ├── core/
│   │   ├── detector.py
│   │   └── warning_engine.py
│   ├── services/
│   │   ├── chatbot.py
│   │   ├── knowledge_base.py
│   │   ├── reporting.py
│   │   └── speech.py
│   └── ui/
│       ├── app.py
│       ├── components.py
│       ├── state.py
│       └── tabs.py
├── detector.py              # wrapper tương thích import cũ
├── database.py              # wrapper tương thích import cũ
├── warning_engine.py        # wrapper tương thích import cũ
├── speech.py                # wrapper tương thích import cũ
├── chatbot.py               # wrapper tương thích import cũ
├── report.py                # wrapper tương thích import cũ
├── models/
│   └── best.pt
├── data/
│   ├── classes.txt
│   ├── signs.json
│   ├── scenarios.json
│   └── penalties.json
├── database/
│   └── app.db
├── assets/
│   ├── test_images/
│   ├── test_videos/
│   └── audio/
├── utils/
│   └── config.py
├── yolo_demo/
│   └── bản FastAPI demo cũ
├── requirements.txt
└── README.md
```

Logic chính được tổ chức trong package `traffic_sign_app/`. Các file `detector.py`, `database.py`, `warning_engine.py`, `speech.py`, `chatbot.py`, `report.py` ở root chỉ re-export hàm để giữ tương thích với yêu cầu đồ án và code cũ.

## Cài đặt

```bash
cd CS338.Q23-TrafficSign-Detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Đặt model

Ứng dụng mặc định load model tại:

```text
models/best.pt
```

Trong project này, model từ `trained_weights/best.pt` đã được copy sang `models/best.pt`. Nếu bạn thay model khác, hãy đặt file YOLO `.pt` vào đúng vị trí trên hoặc nhập đường dẫn khác trong sidebar của app.

Nếu thiếu model, app vẫn mở và hiển thị hướng dẫn thay vì crash.

## Chạy app

```bash
streamlit run main.py
```

Sau đó mở URL Streamlit hiển thị trên terminal, thường là:

```text
http://localhost:8501
```

## Chức năng

### Detect ảnh

Upload ảnh, chạy YOLO, hiển thị ảnh gốc và ảnh đã vẽ bounding box. Mỗi detection hiển thị class ID, tên biển, confidence, loại biển, ý nghĩa, hành động cần làm, cảnh báo, lỗi thường gặp, mức phạt tham khảo và ví dụ.

### Detect video

Upload video, xử lý từng frame theo khoảng cách N frame để giảm lag, hiển thị frame đã vẽ bounding box và lưu lịch sử có cooldown để tránh trùng liên tục.

### Giải thích biển báo

Toàn bộ kiến thức biển báo nằm trong `data/signs.json`, dùng key là `class_id` dạng chuỗi để tránh lỗi tên tiếng Việt có dấu hoặc ký tự đặc biệt.

### Dữ liệu mức phạt tham khảo

Ứng dụng có file `data/penalties.json` lưu mức phạt tham khảo theo hành vi vi phạm, loại phương tiện và căn cứ pháp lý. Với biển giới hạn tốc độ, app dùng `speed_limit_value` trong `data/signs.json` và tốc độ thực tế người dùng nhập để tính mức vượt quá.

Nguồn dữ liệu hiện dùng:

- Nghị định 168/2024/NĐ-CP
- Hiệu lực từ 01/01/2025
- Ngày kiểm chứng dữ liệu: 2026-05-21

Lưu ý: Thông tin mức phạt chỉ phục vụ học tập và demo đồ án. Mức phạt thực tế phụ thuộc hành vi cụ thể, loại phương tiện, tình tiết vi phạm và quy định pháp luật hiện hành.

### Speech

App tạo file `.mp3` trong `assets/audio/` bằng `edge-tts`, nếu lỗi sẽ thử `gTTS`. Nếu TTS không có internet hoặc lỗi, app vẫn chạy bình thường.

### Chatbot

Chatbot rule-based trả lời theo biển báo vừa detect hoặc biển được chọn thủ công. Có thể hỏi về ý nghĩa, cần làm gì, vi phạm, phạt, cảnh báo, ví dụ hoặc tình huống.

### Quiz

Quiz đọc dữ liệu từ `data/scenarios.json`. Hiện có các tình huống quan trọng như P.102, P.123a, P.123b, P.130, P.131a, P.127*50, P.127*60, W.224, W.225, R.302a, R.303 và I.407a.

### Lịch sử

SQLite lưu lịch sử tại `database/app.db`. Tab lịch sử hiển thị bảng detection, tổng số lần nhận diện, biển xuất hiện nhiều nhất, confidence trung bình, lần nhận diện gần nhất, clear history và export CSV.

## Hạn chế

- Kết quả phụ thuộc chất lượng model YOLO, dữ liệu train, ánh sáng, góc chụp và độ rõ của biển báo.
- Mức phạt trong app chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.
- Cần kiểm chứng lại dữ liệu pháp luật trước khi sử dụng chính thức.
- Một số biển cần biển phụ hoặc tình huống thực tế để xác định chính xác hành vi vi phạm.
- Video trong Streamlit là demo gần realtime, chưa tối ưu như hệ thống camera production.
- App không thay thế hệ thống an toàn thực tế trên xe.

## Hướng phát triển

- Thêm webcam realtime ổn định hơn.
- Cập nhật luật và mức phạt từ nguồn chính thống.
- Tối ưu tốc độ inference, batch frame hoặc dùng GPU.
- Thêm mobile app hoặc giao diện nhẹ hơn cho thiết bị yếu.
- Tích hợp LLM API optional để hỏi đáp tự nhiên hơn, nhưng core app vẫn chạy không cần API key.

## Ghi chú dữ liệu

`data/signs.json` đã có đủ 52 class theo dataset. Một số mô tả được viết theo hướng an toàn và phục vụ học tập; trước khi nộp hoặc demo chính thức, bạn nên kiểm chứng lại các phần luật chi tiết và mức phạt theo văn bản hiện hành.
