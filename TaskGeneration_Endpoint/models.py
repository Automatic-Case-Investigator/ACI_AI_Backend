from django.db import models

class BackupModelEntry(models.Model):
    model_name = models.CharField(max_length=512)
    name = models.CharField(max_length=512)
    file_name = models.CharField(max_length=512)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} - {self.file_name}"