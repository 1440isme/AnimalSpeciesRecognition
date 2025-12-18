from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('inference/', views.inference_page, name='inference'),
    path('benchmark/', views.benchmark, name='benchmark'),
    path('api/inference/', views.api_inference, name='api_inference'),
    path('api/system-metrics/', views.api_system_metrics, name='api_system_metrics'),
    path('api/run-benchmark/', views.api_run_benchmark, name='api_run_benchmark'),
    # New benchmark APIs
    path('api/datasets/', views.api_list_datasets, name='api_list_datasets'),
    path('api/upload-dataset/', views.api_upload_dataset, name='api_upload_dataset'),
    path('api/full-benchmark/', views.api_run_full_benchmark, name='api_run_full_benchmark'),
    # Multi-model inference
    path('api/multi-inference/', views.api_multi_inference, name='api_multi_inference'),
]
