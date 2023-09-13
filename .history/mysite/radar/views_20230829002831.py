from django.shortcuts import render

from radar.views_sub import create_radar_chart as crc


def index(request):
    top = crc.TopRadar('LCK', 'Summer', 4)
    top.create_radar()
    return render(request, "radar/index.html")