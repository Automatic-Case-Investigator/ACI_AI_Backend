from django.urls import path
from .objects.view_objects.task_generation_manager import *

urlpatterns = [
    path("history/", HistoryManager.as_view()),
    path("backup/", BackupManager.as_view()),
    path("rollback/", RollbackManager.as_view()),
    path("generate/", TaskGenerationManager.as_view()),
    path("case_tmp_storage/", CaseTemporaryStorageManager.as_view()),
    path("train_model/", TaskGenTrainerManager.as_view()),
    path("restore_baseline/", RestoreManager.as_view()),
]