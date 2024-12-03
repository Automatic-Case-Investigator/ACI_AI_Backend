from django.urls import path
from .objects.view_objects.task_generation_manager import *

urlpatterns = [
    path("generate/", TaskGenerationManager.as_view()),
    path("case_tmp_storage/", CaseTemporaryStorageManager.as_view()),
    path("train_model/", TaskGenTrainerManager.as_view()),
    path("restore_baseline/", RestoreManager.as_view()),
]