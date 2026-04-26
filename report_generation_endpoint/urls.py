from django.urls import path
from .objects.view_objects.report_generation_view import *

urlpatterns = [
    path("generate_activity_report/", ActivityReportGenerator.as_view()),
    path("generate_task_report/", TaskReportGenerator.as_view()),
    path("generate_case_report/", CaseReportGenerator.as_view()),
]
