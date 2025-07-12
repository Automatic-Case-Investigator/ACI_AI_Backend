from django.db import models


class BackupModelEntry(models.Model):
    model_id = models.CharField(max_length=512)
    name = models.CharField(max_length=512)
    basename = models.CharField(max_length=512)
    file_name = models.CharField(max_length=512)
    date_created = models.DateTimeField(auto_now_add=True)


class CurrentBackupModelEntry(models.Model):
    current_model = models.ForeignKey(
        BackupModelEntry,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    model_id = models.CharField(max_length=512)
    name = models.CharField(max_length=512)
