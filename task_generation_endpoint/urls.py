from django.urls import path
from .objects.view_objects.task_generation_view import *

urlpatterns = [
    path("history/", HistoryView.as_view()),
    path("current_backup_version/", CurrentBackupVersionView.as_view()),
    path("current_model_id/", CurrentModelIdView.as_view()),
    path("backup/", BackupView.as_view()),
    path("rollback/", RollbackView.as_view()),
    path("generate/", TaskGenerationView.as_view()),
    path("case_tmp_storage/", CaseTemporaryStorageView.as_view()),
    path("train_model/", TaskGenTrainerView.as_view()),
    path("restore_baseline/", RestoreView.as_view()),
]