from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("radar/", include("radar.urls")),
    path("admin/", admin.site.urls),
]