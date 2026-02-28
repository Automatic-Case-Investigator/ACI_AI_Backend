from django.urls import path
from .objects.view_objects.relevency_filter_view import *

urlpatterns = [
    path("generate/", RelevencyFilterView.as_view()),
]