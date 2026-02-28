from django.urls import path
from .objects.view_objects.summary_view import *

urlpatterns = [
    path("generate_query_summary/", QuerySummaryView.as_view()),
    path("generate_activity_summary/", ActivitySummaryView.as_view()),
]