from django.urls import path
from . import views

urlpatterns = [
    path("add_case_data/", views.add_case_data, name="add_case_data"),
    path("set_case_data/", views.set_case_data, name="set_case_data"),
    path("delete_case_data/", views.delete_case_data, name="delete_case_data"),
    path("train_model/", views.train_model, name="train_model"),
]