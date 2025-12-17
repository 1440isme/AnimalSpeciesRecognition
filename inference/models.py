from django.db import models

class MLModel(models.Model):
    """Model to store ML model information"""
    MODEL_TYPES = [
        ('classification', 'Classification'),
        ('detection', 'Object Detection'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    onnx_path = models.CharField(max_length=500)  # Path to ONNX file
    description = models.TextField(blank=True)
    is_ready = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"
