# 🐾 AI Hub - Animal Species Recognition Platform

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Django](https://img.shields.io/badge/django-5.2-brightgreen.svg)
![ONNX](https://img.shields.io/badge/onnx-runtime-orange.svg)

**Nền tảng AI nhận diện loài động vật với ONNX models, hỗ trợ Classification & Detection**

[Demo](#-demo) • [Cài đặt](#-cài-đặt) • [Tính năng](#-tính-năng) • [API](#-api-endpoints) • [Models](#-models)

</div>

---

## 📸 Demo

| Dashboard | Inference Lab | Benchmark Studio |
|-----------|---------------|------------------|
| Model cards, system metrics | Single/Compare mode | Full dataset evaluation |

---

## 📁 Cấu trúc Project

```
AnimalSpeciesRecognition/
├── aihub/                    # Django project settings
│   ├── settings.py          # Cấu hình Django
│   ├── urls.py              # Main URL routing
│   └── wsgi.py              # WSGI config
├── inference/                # Django app chính
│   ├── views.py             # Views & API endpoints
│   ├── urls.py              # App URL routing
│   └── model_config.py      # Cấu hình models
├── templates/                # HTML templates
│   ├── page-dashboard.html  # Dashboard - Model Hub
│   ├── page-inference.html  # Inference Lab
│   └── page-benchmark.html  # Benchmarking Studio
├── models/                   # ONNX models
│   ├── yolov8m.onnx         # YOLOv8m Detection
│   ├── yolov8m-cls.onnx     # YOLOv8m Classification
│   ├── vit.onnx             # Vision Transformer
│   ├── vgg19.onnx           # VGG19
│   ├── resnet50.onnx        # ResNet50
│   └── efficientnet_b0.onnx # EfficientNet-B0
├── datasets/                 # Benchmark datasets
│   └── (your_datasets)/     # Format: class_name/images/
├── utils_onnx.py            # ONNX inference utilities
├── manage.py                # Django management
└── README.md                # Documentation
```

---

## ✨ Tính năng

### 🏠 Dashboard (`/dashboard/`)
- **Model Hub**: Hiển thị tất cả models với cards đẹp
- **Quick Actions**: Click để Inference hoặc Benchmark
- **System Monitor**: Real-time CPU/GPU usage
- **Search & Filter**: Tìm kiếm theo tên, lọc theo loại model
- **Status Badges**: Ready/Loading cho từng model

### 🔬 Inference Lab (`/inference/`)

#### Single Model Mode
- Chọn model từ dropdown
- Upload ảnh hoặc sử dụng Camera real-time
- Cài đặt Confidence Threshold, Top-K, IoU (detection)
- Kết quả:
  - **Classification**: Top-K predictions với confidence bars
  - **Detection**: Annotated image với bounding boxes

#### Compare Mode
- So sánh 2-6 models cùng lúc
- **Ensemble Voting**: Kết quả bình chọn từ tất cả models
- Biểu đồ so sánh confidence
- Stats: Fastest model, Average time, Success rate

### 📊 Benchmarking Studio (`/benchmark/`)
- **Dataset Management**: Upload hoặc chọn dataset có sẵn
- **Multi-model Benchmark**: Chạy nhiều models cùng lúc
- **Full Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Inference time, FPS
  - Confusion Matrix
- **Visualizations**:
  - Accuracy Comparison Chart
  - Radar Chart (multi-dimensional)
  - Latency Chart
  - Throughput (FPS) Chart
- **Export**: Download kết quả CSV
- **Auto-save**: Lưu kết quả vào localStorage

### 🎨 UI/UX Features
- **Dark/Light Theme**: Toggle ở sidebar, lưu preference
- **Responsive Design**: Desktop & Mobile
- **Smooth Animations**: Transitions mượt mà
- **Real-time Updates**: System metrics, inference progress
- **Modern Design**: Tailwind CSS, Glass morphism

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/1440isme/AnimalSpeciesRecognition.git
cd AnimalSpeciesRecognition
```

### 2. Tạo Virtual Environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt Dependencies

```bash
pip install django onnxruntime-gpu numpy pillow opencv-python psutil GPUtil
```

> **Note**: Dùng `onnxruntime` thay vì `onnxruntime-gpu` nếu không có NVIDIA GPU

### 4. Chuẩn bị Models

Đặt các file ONNX vào thư mục `models/`:

| Model | File | Loại |
|-------|------|------|
| YOLOv8m | `yolov8m.onnx` | Detection |
| YOLOv8m-cls | `yolov8m-cls.onnx` | Classification |
| ViT | `vit.onnx` | Classification |
| VGG19 | `vgg19.onnx` | Classification |
| ResNet50 | `resnet50.onnx` | Classification |
| EfficientNet-B0 | `efficientnet_b0.onnx` | Classification |

### 5. Chạy Server

```bash
python manage.py runserver
```

Mở trình duyệt: **http://127.0.0.1:8000/**

---

## 🔌 API Endpoints

### Inference API

#### `POST /api/inference/`

Chạy inference với single model.

**Request (Form Data):**
```
model_id: string (required)
image: File (required)
confidence_threshold: float (default: 0.5)
top_k: int (default: 5) - for classification
iou_threshold: float (default: 0.45) - for detection
max_detections: int (default: 100) - for detection
```

**Response (Classification):**
```json
{
  "inference_time_ms": 24.5,
  "model_name": "ViT (Vision Transformer)",
  "model_type": "classification",
  "predictions": [
    {"class_id": 4, "class_name": "dog", "confidence": 95.3, "below_threshold": false},
    {"class_id": 1, "class_name": "cat", "confidence": 2.1, "below_threshold": true}
  ],
  "top_prediction": {"class_id": 4, "class_name": "dog", "confidence": 95.3}
}
```

**Response (Detection):**
```json
{
  "inference_time_ms": 32.1,
  "model_name": "YOLOv8m",
  "model_type": "detection",
  "num_detections": 3,
  "detections": [
    {"class_id": 4, "class_name": "dog", "confidence": 87.5, "bbox": [120, 50, 340, 280]}
  ],
  "annotated_image": "base64_encoded_image"
}
```

---

#### `POST /api/multi-inference/`

So sánh nhiều models trên cùng một ảnh.

**Request (Form Data):**
```
model_ids: string (comma-separated, e.g., "vit-classify,resnet50-classify")
image: File (required)
confidence_threshold: float
top_k: int
```

**Response:**
```json
{
  "success": true,
  "model_results": [...],
  "ensemble": {
    "predicted_class": "dog",
    "vote_count": 4,
    "total_models": 5,
    "average_confidence": 89.2,
    "agreement_percentage": 80.0
  },
  "stats": {
    "fastest_model": "EfficientNetB0",
    "fastest_time_ms": 12.3,
    "average_time_ms": 25.6
  }
}
```

---

#### `POST /api/full-benchmark/`

Chạy benchmark đầy đủ trên dataset.

**Request (JSON):**
```json
{
  "dataset_id": "animals-10",
  "model_ids": ["vit-classify", "resnet50-classify"],
  "max_images_per_class": 50
}
```

**Response:**
```json
{
  "success": true,
  "dataset": {
    "id": "animals-10",
    "name": "Animals 10",
    "total_images": 500,
    "num_classes": 10,
    "class_names": ["butterfly", "cat", ...]
  },
  "results": [
    {
      "model_name": "ViT (Vision Transformer)",
      "accuracy": 94.2,
      "precision": 93.8,
      "recall": 94.1,
      "f1_score": 93.9,
      "avg_inference_ms": 28.5,
      "fps": 35.1,
      "confusion_matrix": [[...]]
    }
  ]
}
```

---

#### `GET /api/system-metrics/`

Lấy thông tin CPU/GPU real-time.

**Response:**
```json
{
  "cpu_percent": 45.2,
  "gpu": {
    "name": "NVIDIA GeForce RTX 3060",
    "load": 32.5,
    "memory_used": 2.1,
    "memory_total": 12.0,
    "temperature": 55
  }
}
```

---

#### `GET /api/datasets/`

Liệt kê datasets có sẵn.

---

#### `POST /api/upload-dataset/`

Upload dataset mới (ZIP file với cấu trúc: `class_name/images`).

---

## 🧠 Models

### Animals-10 Dataset Classes

Tất cả models được train trên 10 loài động vật:

| ID | Class Name |
|----|------------|
| 0 | butterfly |
| 1 | cat |
| 2 | chicken |
| 3 | cow |
| 4 | dog |
| 5 | elephant |
| 6 | horse |
| 7 | sheep |
| 8 | spider |
| 9 | squirrel |

### Thêm Model Mới

1. **Thêm config** vào `inference/model_config.py`:

```python
{
    'id': 'new-model-id',
    'name': 'Model Display Name',
    'type': 'classification',  # hoặc 'detection'
    'onnx_path': 'models/your_model.onnx',
    'description': 'Mô tả model',
    'is_ready': True,
    'classes': 10,
    'class_names': ['butterfly', 'cat', ...],
    'preprocessing': {
        'resize_method': 'center_crop',  # 'squash', 'letterbox'
        'normalization': 'imagenet',     # 'simple', 'caffe', 'none'
        'input_layout': 'NCHW'           # hoặc 'NHWC'
    }
}
```

2. **Đặt file ONNX** vào `models/`
3. **Restart server**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Django 5.2, Python 3.8+ |
| **AI Runtime** | ONNX Runtime (CPU/CUDA) |
| **Image Processing** | OpenCV, Pillow, NumPy |
| **System Monitoring** | psutil, GPUtil |
| **Frontend** | Tailwind CSS, Chart.js |
| **Icons** | Material Symbols |

---

## 📋 Requirements

```txt
django>=5.0
onnxruntime>=1.16.0  # hoặc onnxruntime-gpu
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0
psutil>=5.9.0
GPUtil>=1.4.0
```

---

## 🎯 Workflow Sử Dụng

### Quick Inference
1. Vào **Dashboard** → Chọn model
2. Click **Inference** → Upload ảnh
3. Xem kết quả với confidence scores

### Compare Models
1. Vào **Inference Lab** → Chọn **Compare Mode**
2. Tick chọn 2+ models
3. Upload ảnh → So sánh kết quả
4. Xem **Ensemble Voting** result

### Full Benchmark
1. Vào **Benchmark Studio**
2. Chọn/Upload dataset
3. Chọn models cần benchmark
4. Chạy benchmark → Xem metrics đầy đủ
5. Export CSV nếu cần

---

## 🔒 Lưu ý

- **CSRF**: API endpoints sử dụng `@csrf_exempt` cho development. Production nên cấu hình CSRF token.
- **GPU**: Ưu tiên CUDA nếu có GPU NVIDIA, tự động fallback CPU.
- **Model Cache**: Models được cache sau lần load đầu tiên để tăng tốc.

---

## 📝 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">
Made with ❤️ for Animal Species Recognition
</div>
