from django.urls import path

from .objects.view_objects.config_file_view import ConfigFileView

urlpatterns = [
    path("config_files/", ConfigFileView.as_view()),
]
