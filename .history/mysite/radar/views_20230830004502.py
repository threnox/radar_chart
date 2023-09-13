from django.shortcuts import render

from radar.views_sub import create_radar_chart as crc
from radar.forms import ParameterForm


def index(request):

    if request.method == 'POST':
        form = ParameterForm(request.POST)
        context = {'form': form}
        league = request.POST['region_choice']
        split = request.POST['season_choice']
        min_game_count = request.POST['min_games']

        # params['forms'] = LFForm(request.POST)
    # league, split, min_game_count = 'LCK', 'Spring', 4
        top = crc.TopRadar(league, split, min_game_count)
        top.create_radar()
    return render(request, 'radar/index.html', context)