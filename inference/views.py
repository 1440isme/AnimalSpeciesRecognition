from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import time
from io import BytesIO
import base64
from utils_onnx import load_model, preprocess_image, run_inference, softmax
import numpy as np
import cv2
from PIL import Image
from .model_config import get_model_by_id, get_all_models

# Global model cache to avoid reloading models on every request
MODEL_CACHE = {}

def dashboard(request):
    """Render dashboard page"""
    models = get_all_models()
    return render(request, 'page-dashboard.html', {'models': models})

def inference_page(request):
    """Render inference page"""
    model_id = request.GET.get('model_id', None)
    model_info = None
    if model_id:
        model_info = get_model_by_id(model_id)
    return render(request, 'page-inference.html', {'model_info': model_info})

def benchmark(request):
    """Render benchmark page"""
    return render(request, 'page-benchmark.html')

@csrf_exempt
def api_inference(request):
    """API endpoint for ONNX inference using cached models for real-time performance"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Get model_id from request
        model_id = request.POST.get('model_id')
        if not model_id:
            return JsonResponse({'error': 'Missing model_id'}, status=400)
        
        # Get inference settings from request (with defaults)
        confidence_threshold = float(request.POST.get('confidence_threshold', 0.5))
        top_k = int(request.POST.get('top_k', 5))
        iou_threshold = float(request.POST.get('iou_threshold', 0.45))
        max_detections = int(request.POST.get('max_detections', 100))
        
        # Get model configuration
        model_config = get_model_by_id(model_id)
        if not model_config:
            return JsonResponse({'error': 'Model not found'}, status=404)
        
        if not model_config['is_ready']:
            return JsonResponse({'error': 'Model is not ready'}, status=400)
        
        # Check if image is present
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Missing image file'}, status=400)
        
        image_file = request.FILES['image']
        
        # Validate file extension
        allowed_image_ext = ('.png', '.jpg', '.jpeg', '.webp')
        if not image_file.name.lower().endswith(allowed_image_ext):
            return JsonResponse({'error': 'Invalid image file type'}, status=400)
        
        # Get model path from config
        model_path = os.path.join(os.path.dirname(__file__), '..', model_config['onnx_path'])
        model_path = os.path.abspath(model_path)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return JsonResponse({'error': f'Model file not found: {model_path}'}, status=500)
        
        # Load model from cache or load new one
        if model_path not in MODEL_CACHE:
            print(f"🔄 Loading model into cache: {model_path}")
            session = load_model(model_path)
            if not session:
                return JsonResponse({'error': 'Failed to load ONNX model'}, status=500)
            MODEL_CACHE[model_path] = session
        else:
            session = MODEL_CACHE[model_path]

        
        # Get input shape and target size
        try:
            input_shape = session.get_inputs()[0].shape
            
            # Determine layout from config or shape
            layout = 'NCHW'
            if 'preprocessing' in model_config and 'input_layout' in model_config['preprocessing']:
                layout = model_config['preprocessing']['input_layout']
            elif len(input_shape) == 4 and input_shape[3] == 3:
                layout = 'NHWC'
            
            if layout == 'NHWC':
                # [Batch, Height, Width, Channels]
                h = input_shape[1] if isinstance(input_shape[1], int) else 224
                w = input_shape[2] if isinstance(input_shape[2], int) else 224
            else:
                # [Batch, Channels, Height, Width]
                h = input_shape[2] if isinstance(input_shape[2], int) else 224
                w = input_shape[3] if isinstance(input_shape[3], int) else 224
        except Exception as e:
            print(f"Error determining input shape: {e}")
            h, w = 224, 224
        
        # Read image
        image_bytes = BytesIO(image_file.read())
        
        # Preprocess image
        preprocessing_config = model_config.get('preprocessing', {})
        input_data = preprocess_image(image_bytes, target_size=(h, w), preprocessing_config=preprocessing_config)
        
        if input_data is None:
            return JsonResponse({'error': 'Failed to preprocess image'}, status=500)
        
        # Run inference
        start_time = time.time()
        outputs = run_inference(session, input_data)
        end_time = time.time()
        
        if outputs is None:
            return JsonResponse({'error': 'Inference failed'}, status=500)
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Process output based on model type
        output_tensor = outputs[0]
        results = {
            'inference_time_ms': round(inference_time, 2),
            'output_shape': list(output_tensor.shape),
            'model_name': model_config['name'],
            'model_type': model_config['type'],
        }
        
        if model_config['type'] == 'classification':
            # Classification: return top-K predictions (user-configurable)
            probs = softmax(output_tensor)
            
            # Get top-K predictions first (sorted by confidence)
            top_k_indices = np.argsort(probs[0])[::-1][:top_k]
            
            predictions = []
            for idx in top_k_indices:
                conf = float(probs[0][idx] * 100)
                pred = {
                    'class_id': int(idx),
                    'confidence': conf
                }
                # Add class name if available
                if 'class_names' in model_config and idx < len(model_config['class_names']):
                    pred['class_name'] = model_config['class_names'][idx]
                
                # Mark if below confidence threshold (for UI indication)
                pred['below_threshold'] = conf < (confidence_threshold * 100)
                
                predictions.append(pred)
            
            results['predictions'] = predictions
            results['top_prediction'] = predictions[0] if predictions else None
            results['confidence_threshold'] = confidence_threshold * 100  # Send threshold to frontend
            
        elif model_config['type'] == 'detection':
            # Detection: process bounding boxes and create annotated image
            # Reset image file pointer
            image_file.seek(0)
            image_bytes_reset = BytesIO(image_file.read())
            
            # Load original image
            pil_image = Image.open(image_bytes_reset)
            original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # NMS (Non-Maximum Suppression) function
            def nms(boxes, scores, iou_threshold=0.45):
                """Apply NMS to filter overlapping boxes"""
                if len(boxes) == 0:
                    return []
                
                # Convert to numpy arrays
                boxes = np.array(boxes)
                scores = np.array(scores)
                
                # Get coordinates
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                
                # Calculate areas
                areas = (x2 - x1) * (y2 - y1)
                
                # Sort by scores
                order = scores.argsort()[::-1]
                
                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)
                    
                    # Calculate IoU with remaining boxes
                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])
                    
                    w = np.maximum(0.0, xx2 - xx1)
                    h = np.maximum(0.0, yy2 - yy1)
                    inter = w * h
                    
                    iou = inter / (areas[i] + areas[order[1:]] - inter)
                    
                    # Keep boxes with IoU less than threshold
                    inds = np.where(iou <= iou_threshold)[0]
                    order = order[inds + 1]
                
                return keep
            
            # Process detection output (YOLOv8 ONNX format)
            # YOLOv8 output shape: [1, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
            # Need to transpose to [1, 8400, 84] for easier processing
            all_boxes = []
            all_scores = []
            all_class_ids = []
            
            # Handle YOLOv8 output format
            if len(output_tensor.shape) == 3:
                # Transpose if needed: [1, 84, 8400] -> [1, 8400, 84]
                if output_tensor.shape[1] < output_tensor.shape[2]:
                    output_tensor = output_tensor.transpose(0, 2, 1)
                
                num_detections = output_tensor.shape[1]
                # Use user-provided confidence threshold
                
                # Get image dimensions
                img_h, img_w = original_image.shape[:2]
                
                # YOLOv8 ONNX model input size (usually 640x640)
                # Need to get actual input size from model
                try:
                    input_shape = session.get_inputs()[0].shape
                    model_input_h = input_shape[2] if len(input_shape) > 2 else 640
                    model_input_w = input_shape[3] if len(input_shape) > 3 else 640
                except:
                    model_input_h = model_input_w = 640
                
                # Calculate scale factors
                scale_x = img_w / model_input_w
                scale_y = img_h / model_input_h
                
                # Collect all detections above threshold
                for i in range(num_detections):
                    detection = output_tensor[0, i, :]
                    
                    # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
                    x_center, y_center, width, height = detection[0:4]
                    class_scores = detection[4:]
                    
                    # Get class with highest score
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
                    
                    if confidence > confidence_threshold:
                        # YOLOv8 ONNX outputs coordinates relative to input size (e.g. 640x640)
                        
                        # Determine resize method and padding
                        resize_method = 'squash'
                        if 'preprocessing' in model_config:
                            resize_method = model_config['preprocessing'].get('resize_method', 'squash')
                        
                        if resize_method == 'letterbox':
                            # Calculate padding and scale used in preprocessing
                            scale = min(model_input_w / img_w, model_input_h / img_h)
                            new_w = int(img_w * scale)
                            new_h = int(img_h * scale)
                            pad_w = (model_input_w - new_w) // 2
                            pad_h = (model_input_h - new_h) // 2
                            
                            # Adjust coordinates to remove padding
                            x_center = x_center - pad_w
                            y_center = y_center - pad_h
                            
                            # Scale back to original size
                            x_center_scaled = x_center / scale
                            y_center_scaled = y_center / scale
                            width_scaled = width / scale
                            height_scaled = height / scale
                            
                        else:
                            # Standard squash scaling
                            x_center_scaled = x_center * scale_x
                            y_center_scaled = y_center * scale_y
                            width_scaled = width * scale_x
                            height_scaled = height * scale_y
                        
                        # Convert to corner coordinates
                        x1 = int(x_center_scaled - width_scaled/2)
                        y1 = int(y_center_scaled - height_scaled/2)
                        x2 = int(x_center_scaled + width_scaled/2)
                        y2 = int(y_center_scaled + height_scaled/2)
                        
                        # Clip to image boundaries
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(confidence)
                        all_class_ids.append(class_id)
                
                # Apply NMS to filter overlapping boxes (use user-provided IoU threshold)
                keep_indices = nms(all_boxes, all_scores, iou_threshold=iou_threshold)
                
                # Draw only the boxes that passed NMS (limit to max_detections)
                detections = []
                for i, idx in enumerate(keep_indices):
                    if i >= max_detections:
                        break
                    
                    x1, y1, x2, y2 = all_boxes[idx]
                    confidence = all_scores[idx]
                    class_id = all_class_ids[idx]
                    
                    # Get class name
                    class_name = f"Class {class_id}"
                    if 'class_names' in model_config and class_id < len(model_config['class_names']):
                        class_name = model_config['class_names'][class_id]
                    
                    # Generate distinct color for each class (using HSV color space)
                    # This creates vibrant, distinguishable colors
                    import colorsys
                    num_classes = model_config.get('classes', 80)
                    hue = (class_id * 0.618033988749895) % 1.0  # Golden ratio for better distribution
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)  # High saturation and value
                    box_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
                    
                    # Draw bounding box with class-specific color
                    box_thickness = 2
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), box_color, box_thickness)
                    
                    # Prepare label
                    label = f"{class_name} {confidence:.2f}"
                    
                    # Calculate adaptive font scale based on bbox size
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    img_area = img_w * img_h
                    
                    # Adaptive font scale (smaller for small boxes, larger for big boxes)
                    if bbox_area < img_area * 0.01:  # Very small box
                        font_scale = 0.3
                        font_thickness = 1
                    elif bbox_area < img_area * 0.05:  # Small box
                        font_scale = 0.4
                        font_thickness = 1
                    elif bbox_area < img_area * 0.15:  # Medium box
                        font_scale = 0.5
                        font_thickness = 1
                    else:  # Large box
                        font_scale = 0.6
                        font_thickness = 2
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                    
                    # Determine label position (above or inside bbox)
                    label_y = y1 - 5  # Default: above bbox
                    label_bg_y1 = y1 - text_height - baseline - 8
                    label_bg_y2 = y1 - 2
                    
                    # If label would be cut off at top, put it inside bbox
                    if label_bg_y1 < 0:
                        label_y = y1 + text_height + 5
                        label_bg_y1 = y1 + 2
                        label_bg_y2 = y1 + text_height + baseline + 8
                    
                    # If label would be cut off at right, adjust x position
                    label_x = x1 + 3
                    label_bg_x2 = x1 + text_width + 8
                    if label_bg_x2 > img_w:
                        label_x = max(0, img_w - text_width - 8)
                        label_bg_x2 = img_w
                    
                    # Draw background for text
                    text_bg_color = box_color  # Same color as box
                    cv2.rectangle(original_image, 
                                (label_x - 3, label_bg_y1), 
                                (label_bg_x2, label_bg_y2), 
                                text_bg_color, -1)
                    
                    # Draw text
                    text_color = (255, 255, 255)  # White
                    cv2.putText(original_image, label, 
                              (label_x, label_y), 
                              font, font_scale, text_color, font_thickness)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence * 100,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', original_image)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            results['detections'] = detections
            results['num_detections'] = len(detections)
            results['annotated_image'] = annotated_image_base64
        
        return JsonResponse(results, status=200)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def api_system_metrics(request):
    """API endpoint to get real-time system metrics"""
    try:
        import psutil
        import GPUtil
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get GPU usage
        gpu_info = {
            'name': 'N/A',
            'load': 0,
            'memory_used': 0,
            'memory_total': 0
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get first GPU
                gpu_info = {
                    'name': gpu.name,
                    'load': round(gpu.load * 100, 1),
                    'memory_used': round(gpu.memoryUsed / 1024, 2),  # Convert to GB
                    'memory_total': round(gpu.memoryTotal / 1024, 2),
                    'temperature': gpu.temperature
                }
        except:
            pass  # No GPU or GPUtil not working
        
        return JsonResponse({
            'cpu_percent': round(cpu_percent, 1),
            'gpu': gpu_info
        }, status=200)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
