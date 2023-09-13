from django.http import HttpResponse
from django.http import Http404
from django.shortcuts import render
from django.template import loader
from views_sub import create_radar_chart as crc


def index(request):
    top = crc.TopRadar('LCK', 'Summer', 4)
    top.create_radar()
    return render(request, "radar/index.html")