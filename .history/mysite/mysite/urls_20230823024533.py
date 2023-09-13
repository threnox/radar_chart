from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("radar/", include("radar.urls.test")),
    path("admin/", admin.site.urls),
]