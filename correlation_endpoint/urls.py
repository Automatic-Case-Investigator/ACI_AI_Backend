from django.urls import path
from .objects.view_objects.correlation_view import CorrelationView

urlpatterns = [
    path("query/", CorrelationView.as_view())
]