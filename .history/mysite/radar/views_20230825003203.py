from django.http import HttpResponse
from django.shortcuts import render
from views_sub import create_radar_chart as crc

def index(request):
    top = crc.TopRadar('LCK', 'Summer', 4)
    return HttpResponse(top.create_radar())