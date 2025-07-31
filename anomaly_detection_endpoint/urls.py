from django.urls import path
from .objects.view_objects.anomaly_detection_view import AnomalyDetectionView

urlpatterns = [
    path("query/", AnomalyDetectionView.as_view())
]