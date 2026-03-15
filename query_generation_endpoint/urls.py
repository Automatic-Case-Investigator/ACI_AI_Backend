from django.urls import path
from .objects.view_objects.fix_queries_view import *
from .objects.view_objects.query_generation_view import *

urlpatterns = [
    path("fix_queries/", FixQueriesView.as_view()),
    path("generate/", QueryGenerationView.as_view()),
]
