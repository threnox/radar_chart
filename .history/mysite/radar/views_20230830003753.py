from django.shortcuts import render

from .views_sub import create_radar_chart as crc
from .forms import ParameterForm


def index(request):
    form = ParameterForm()
    context = {'form': form}
    # if request.method == 'POST':
    #     league = request.POST['region_choice']
    #     split = request.POST['season_choice']
    #     min_game_count = request.POST['min_games']

    #     # params['forms'] = LFForm(request.POST)
    #     top = crc.TopRadar(league, split, min_game_count)
    #     top.create_radar()
    return render(request, 'radar/index.html', context)