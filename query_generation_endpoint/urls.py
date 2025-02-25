from django.urls import path
from .objects.view_objects.query_generation_view import *

urlpatterns = [
    path("generate/", QueryGenerationView.as_view()),
    path("restore_baseline/", RestoreView.as_view()),
]