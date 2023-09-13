from django.urls import path

from . import views

app_name = 'radar'

urlpatterns = [
    path("", views.generator, name="index"),
    path("display", views.display, name="display"),
]