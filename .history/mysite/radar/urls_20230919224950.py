from django.urls import path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from . import views

app_name = 'radar'

urlpatterns = [
    path("", views.generator, name="index"),
    path("display", views.display, name="display"),
]