"""
Configuration for pre-trained models
This file contains the list of available models in the system
"""

MODELS_CONFIG = [
    {
        'id': 'yolov8m-detect',
        'name': 'YOLOv8m',
        'type': 'detection',
        'onnx_path': 'models/yolov8m.onnx',
        'description': 'State-of-the-art object detection model optimized for real-time inference speed and accuracy balance.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'letterbox',  # Detection usually works best with letterbox (pad to square)
            'normalization': 'simple'      # 0-1
        }
    },
    {
        'id': 'vit-classify',
        'name': 'ViT (Vision Transformer)',
        'type': 'classification',
        'onnx_path': 'models/vit.onnx',
        'description': 'Transformer architecture applied directly to sequences of image patches for high-accuracy classification.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'center_crop',
            'normalization': 'imagenet'
        }
    },
    {
        'id': 'vgg19-classify',
        'name': 'VGG19',
        'type': 'classification',
        'onnx_path': 'models/vgg19.onnx',
        'description': 'A classic convolutional neural network model proposed by K. Simonyan and A. Zisserman from Oxford.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'center_crop',
            'normalization': 'imagenet',
            'input_layout': 'NHWC'  # Model ONNX expects NHWC format
        }
    },
    {
        'id': 'resnet50-classify',
        'name': 'ResNet50',
        'type': 'classification',
        'onnx_path': 'models/resnet50.onnx',
        'description': 'Deep residual networks that allow training of much deeper networks by using skip connections.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'center_crop',
            'normalization': 'caffe',  # Caffe-style: BGR + mean subtraction
            'input_layout': 'NHWC'  # Model ONNX expects NHWC format
        }
    },
    {
        'id': 'efficientnet-classify',
        'name': 'EfficientNetB0',
        'type': 'classification',
        'onnx_path': 'models/efficientnet_b0.onnx',
        'description': 'Highly efficient convolutional neural networks that scale depth, width, and resolution.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'squash',
            'normalization': 'none',  # Model ONNX đã bao gồm preprocessing
            'input_layout': 'NHWC'
        }
    },
    {
        'id': 'yolov8m-cls',
        'name': 'YOLOv8m-cls',
        'type': 'classification',
        'onnx_path': 'models/yolov8m-cls.onnx',
        'description': 'Adaptation of the YOLOv8 architecture specifically tuned for image classification tasks.',
        'is_ready': True,
        'classes': 10,  # animal-10 dataset
        'class_names': ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel'],
        'preprocessing': {
            'resize_method': 'center_crop', # Torchvision/YOLO-cls usually uses center crop for eval
            'normalization': 'simple'       # YOLO usually uses 0-1
        }
    },
]

def get_model_by_id(model_id):
    """Get model configuration by ID"""
    for model in MODELS_CONFIG:
        if model['id'] == model_id:
            return model
    return None

def get_all_models():
    """Get all models configuration"""
    return MODELS_CONFIG
