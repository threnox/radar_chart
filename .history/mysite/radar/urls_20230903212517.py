from django.urls import path

from . import views

urlpatterns = [
    path("", views.generator, name="index"),
    path("display", views.generator, name="display"),
]