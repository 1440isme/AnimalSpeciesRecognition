import onnxruntime as ort
import numpy as np
from PIL import Image
import io


def load_model(model_path):
    """
    Load an ONNX model session with GPU acceleration (CUDA) if available.
    Falls back to CPU if GPU is not available.
    """
    try:
        # Get available providers
        available_providers = ort.get_available_providers()
        
        # Prioritize CUDA (GPU) over CPU
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print(f"✓ Using GPU (CUDA) for inference: {model_path}")
        else:
            print(f"⚠ GPU not available, using CPU for inference: {model_path}")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        # Session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Parallel execution within ops
        sess_options.inter_op_num_threads = 4  # Parallel execution between ops
        
        # Create session with optimized settings
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # Log which provider is actually being used
        active_provider = session.get_providers()[0]
        print(f"→ Active provider: {active_provider}")
        
        return session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None


def preprocess_image(image_bytes, target_size=(224, 224), preprocessing_config=None):
    """
    Preprocess image for generic classification models.
    Supports:
    - Resize methods: 'squash' (default), 'center_crop', 'letterbox'
    - Normalization: 'none' (0-255), 'simple' (0-1), 'imagenet' (mean/std)
    - Input layout: 'NCHW' (default), 'NHWC'
    """
    if preprocessing_config is None:
        preprocessing_config = {}
        
    resize_method = preprocessing_config.get('resize_method', 'squash')
    normalization = preprocessing_config.get('normalization', 'simple')
    
    try:
        image = Image.open(image_bytes).convert("RGB")
        
        # 1. Resize strategy
        if resize_method == 'center_crop':
            # Resize shortest side to target_size, then center crop
            # This is standard for ResNet, VGG, ViT, etc.
            img_w, img_h = image.size
            target_h, target_w = target_size
            
            scale = max(target_w / img_w, target_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            image = image.resize((new_w, new_h), Image.BILINEAR)
            
            # Center crop
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            image = image.crop((left, top, left + target_w, top + target_h))
            
        elif resize_method == 'letterbox':
            # Resize longest side to target_size, pad the rest
            # Standard for YOLO detection
            img_w, img_h = image.size
            target_h, target_w = target_size
            
            scale = min(target_w / img_w, target_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            image = image.resize((new_w, new_h), Image.BILINEAR)
            
            # Create new background image
            new_image = Image.new("RGB", (target_w, target_h), (114, 114, 114))
            
            # Paste resized image in center
            left = (target_w - new_w) // 2
            top = (target_h - new_h) // 2
            new_image.paste(image, (left, top))
            image = new_image
            
        else:
            # Default 'squash'
            image = image.resize(target_size)
            
        # 2. Convert to Array and Normalize
        img_data = np.array(image).astype("float32")
        
        # Check input layout (NCHW vs NHWC)
        input_layout = preprocessing_config.get('input_layout', 'NCHW')
        
        if input_layout == 'NCHW':
            img_data = np.transpose(img_data, (2, 0, 1))  # HWC -> CHW
            img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension -> NCHW
        else:
            # NHWC (Keep HWC, just add batch)
            img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension -> NHWC
        
        
        # 3. Specific Normalization
        if normalization == 'none':
            # No normalization - keep as 0-255 range
            # Some ONNX models already include preprocessing
            pass
        elif normalization == 'simple':
            # Simple 0-1 normalization
            img_data /= 255.0
        elif normalization == 'imagenet':
            # Standard ImageNet normalization: (x - mean) / std
            img_data /= 255.0  # First normalize to 0-1
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]
            
            if input_layout == 'NCHW':
                mean = np.array(mean_vals).reshape((1, 3, 1, 1))
                std = np.array(std_vals).reshape((1, 3, 1, 1))
            else:
                # NHWC
                mean = np.array(mean_vals).reshape((1, 1, 1, 3))
                std = np.array(std_vals).reshape((1, 1, 1, 3))
                
            img_data = (img_data - mean) / std
            
        return img_data.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def run_inference(session, image_data):
    """
    Run inference on the preprocessed image data.
    """
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name], {input_name: image_data})
        return result
    except Exception as e:
        print(f"Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    
    # Handle inputs that are already probabilities (e.g. from models with Softmax layer)
    # If sum is close to 1.0, return as is
    if x.ndim == 2:
        row_sums = np.sum(x, axis=1)
        if np.allclose(row_sums, 1.0, atol=1e-2) and np.all(x >= 0) and np.all(x <= 1.01):
            return x
            
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
