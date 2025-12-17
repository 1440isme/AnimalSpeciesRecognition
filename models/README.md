# Models Directory

This directory contains ONNX model files for inference.

## Required Models

Place your trained ONNX models in this directory with the following names:

### Classification Models
- `yolov8m-cls.onnx` - YOLOv8m Classification model (trained on animal-10 dataset)
- `vit.onnx` - Vision Transformer
- `vgg16.onnx` - VGG16
- `resnet50.onnx` - ResNet50
- `efficientnet_b0.onnx` - EfficientNet B0

### Detection Models
- `yolov8m.onnx` - YOLOv8m Object Detection model

## Converting Models to ONNX

### YOLOv8 Models
```python
from ultralytics import YOLO

# For classification model
model = YOLO('path/to/your/best.pt')
model.export(format='onnx', simplify=True)

# For detection model
model = YOLO('yolov8m.pt')
model.export(format='onnx', simplify=True)
```

### PyTorch Models
```python
import torch

# Load your model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Model Configuration

Models are configured in `inference/model_config.py`. Update the configuration if you add new models or change paths.
