from django.urls import path
from .objects.view_objects.activity_generation_view import *

urlpatterns = [
    path("generate/", ActivityGenerationView.as_view()),
]