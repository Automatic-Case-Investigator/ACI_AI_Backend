from django.urls import path
from .objects.view_objects.task_generation_view import *

urlpatterns = [
    path("generate/", TaskGenerationView.as_view()),
]
