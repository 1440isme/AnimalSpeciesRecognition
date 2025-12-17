# AI Hub - ONNX Model Inference Platform

Web application để chạy inference với ONNX models, được xây dựng bằng Django.

## 📁 Cấu trúc Project

```
WebAIHub/
├── aihub/              # Django project settings
│   ├── settings.py    # Cấu hình Django
│   ├── urls.py        # Main URL routing
│   └── wsgi.py        # WSGI config
├── inference/          # Django app chính
│   ├── views.py       # Views xử lý requests
│   ├── urls.py        # App URL routing
│   └── model_config.py # Cấu hình models
├── templates/          # HTML templates (UI của bạn)
│   ├── page-dashboard.html    # ✅ Dashboard với model cards
│   ├── page-inference.html    # ✅ Inference page
│   └── page-benchmark.html    # ✅ Benchmark results
├── models/             # 📦 Thư mục chứa ONNX models
│   ├── yolov8m-cls.onnx      # Classification model (animal-10)
│   ├── yolov8m.onnx          # Detection model
│   └── README.md             # Hướng dẫn models
├── utils_onnx.py      # ONNX inference utilities
├── convert_to_onnx.py # Script convert models sang ONNX
├── manage.py          # Django management script
├── db.sqlite3         # SQLite database
└── README.md          # File này
```

## ✨ Tính năng

### 🎯 Hệ thống hoạt động với Pre-loaded Models

**Nguyên tắc**: Hệ thống sử dụng tài nguyên máy người dùng, models được lưu sẵn trên hệ thống.

### 1. **Dashboard** (`/` hoặc `/dashboard/`)
- ✅ Hiển thị danh sách models từ `model_config.py`
- ✅ Model cards động với thông tin:
  - Tên model
  - Loại (Classification/Detection)
  - Trạng thái (Ready/Loading)
  - Description
- ✅ Sidebar navigation (Dashboard, Inference, Benchmark History)
- ✅ **Real-time System Status** (CPU/GPU load từ máy thật)
  - Auto-refresh mỗi 2 giây
  - Hiển thị tên GPU thực tế
  - Progress bars động
- ✅ Click "Inference" → chuyển đến `/inference/?model_id=xxx`

### 2. **Inference Lab** (`/inference/?model_id=xxx`)
- ✅ **Nhận model_id từ dashboard** qua URL parameter
- ✅ **Hiển thị thông tin model đã chọn**
- ✅ **Chỉ upload ảnh** (không upload model)
- ✅ **Run Inference** với model từ hệ thống
- ✅ **Kết quả theo loại model**:

  **Classification Models:**
  - Top 5 predictions với class names (nếu có)
  - Confidence scores với progress bars
  - Highlight prediction cao nhất
  - Class names cho animal-10 dataset

  **Detection Models:**
  - Ảnh đã vẽ bounding boxes
  - Danh sách objects detected
  - Confidence scores
  - Bounding box coordinates

### 3. **Benchmark Results** (`/benchmark/`)
- ✅ Top navigation bar
- ✅ Summary stats và charts
- ✅ Detailed metrics table

### 4. **Backend API**

#### `/api/inference/` (POST)
**Request:**
```javascript
formData.append('model_id', 'yolov8m-cls');  // Model ID từ config
formData.append('image', imageFile);          // File object
```

**Response (Classification):**
```json
{
  "inference_time_ms": 24.5,
  "output_shape": [1, 10],
  "model_name": "YOLOv8m-cls",
  "model_type": "classification",
  "predictions": [
    {"class_id": 0, "class_name": "butterfly", "confidence": 95.3},
    {"class_id": 1, "class_name": "cat", "confidence": 2.1},
    ...
  ],
  "top_prediction": {"class_id": 0, "class_name": "butterfly", "confidence": 95.3}
}
```

**Response (Detection):**
```json
{
  "inference_time_ms": 32.1,
  "output_shape": [1, 25200, 85],
  "model_name": "YOLOv8m",
  "model_type": "detection",
  "num_detections": 3,
  "detections": [
    {"class_id": 16, "confidence": 87.5, "bbox": [120, 50, 340, 280]},
    ...
  ],
  "annotated_image": "base64_encoded_image_with_bboxes"
}
```

#### `/api/system-metrics/` (GET)
- Real-time CPU/GPU metrics
- Auto-refresh mỗi 2 giây

## 🚀 Setup và Chạy ứng dụng

### 1. Cài đặt dependencies

```bash
pip install django onnxruntime numpy pillow psutil GPUtil opencv-python
```

### 2. Chuẩn bị ONNX Models

**Cách 1: Sử dụng script tự động**
```bash
# Đặt file .pt models vào thư mục models/
# Sau đó chạy:
python convert_to_onnx.py
```

**Cách 2: Convert thủ công**
```python
from ultralytics import YOLO

# Classification model (animal-10)
model = YOLO('path/to/your/best.pt')
model.export(format='onnx', simplify=True)
# Đổi tên thành yolov8m-cls.onnx và đặt vào models/

# Detection model
model = YOLO('yolov8m.pt')
model.export(format='onnx', simplify=True)
# Đổi tên thành yolov8m.onnx và đặt vào models/
```

### 3. Chạy server

```bash
python manage.py runserver
```

Mở: **http://127.0.0.1:8000/**

## 🎯 Workflow sử dụng

1. **Vào Dashboard** → Xem danh sách models có sẵn
2. **Click "Inference"** trên model card → Chuyển đến inference page với model đã chọn
3. **Upload ảnh test** (JPG/PNG/WEBP)
4. **Click "Run Inference"**
5. **Xem kết quả**:
   - **Classification**: Top 5 predictions với class names
   - **Detection**: Ảnh với bounding boxes + danh sách objects

## � Cấu hình Models

Models được cấu hình trong `inference/model_config.py`:

```python
MODELS_CONFIG = [
    {
        'id': 'yolov8m-cls',
        'name': 'YOLOv8m-cls',
        'type': 'classification',
        'onnx_path': 'models/yolov8m-cls.onnx',
        'description': '...',
        'is_ready': True,
        'classes': 10,
        'class_names': ['butterfly', 'cat', 'chicken', ...]
    },
    ...
]
```

Để thêm model mới:
1. Thêm config vào `MODELS_CONFIG`
2. Đặt file ONNX vào thư mục `models/`
3. Restart server

## 🛠️ Dependencies

- Django 5.2.8
- onnxruntime 1.23.2
- numpy
- Pillow
- opencv-python (cv2) - Vẽ bounding boxes
- psutil - Real-time CPU monitoring
- GPUtil - Real-time GPU monitoring

## 🎨 UI Features

- **100% UI từ HTML templates** của bạn
- **Dynamic model cards** từ backend config
- **Dark theme** với Tailwind CSS
- **Responsive design**
- **Real-time feedback**: loading states, error messages
- **Conditional rendering** dựa trên model type

## 🚧 Tính năng có thể mở rộng

- [ ] Lưu lịch sử inference vào database
- [ ] Thêm authentication/user management
- [ ] Benchmark runner thực tế
- [ ] Export results to CSV/JSON
- [ ] Batch inference
- [ ] WebSocket cho real-time updates
- [ ] Model upload/management UI
- [ ] Support thêm model types (segmentation, pose estimation)
