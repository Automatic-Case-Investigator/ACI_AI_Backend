from django.urls import path
from .objects.view_objects.completion_check_view import *

urlpatterns = [
    path("activity_completion_check/", ActivityCompletionCheckView.as_view()),
]