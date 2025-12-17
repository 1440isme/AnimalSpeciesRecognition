from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('inference/', views.inference_page, name='inference'),
    path('benchmark/', views.benchmark, name='benchmark'),
    path('api/inference/', views.api_inference, name='api_inference'),
    path('api/system-metrics/', views.api_system_metrics, name='api_system_metrics'),
]
