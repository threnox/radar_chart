from django.shortcuts import render

from .views_sub import create_radar_chart as crc
from . import forms


def index(request):
    top = crc.TopRadar('LCK', 'Summer', 4)
    top.create_radar()
    return render(request, "radar/index.html", {'form': forms.ParameterForm()})