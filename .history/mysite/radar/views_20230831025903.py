from django.shortcuts import render

from .views_sub import create_radar_chart as crc
from .forms import ParameterForm


def index(request):

    # if request.method == 'POST':
    


    # league, split, min_game_count = 'LCK', 'Summer', 4
        # top = crc.TopRadar('LCK', 'Summer', 4)
    if request.method == "POST":
        
        if 'generate' in request.POST:
            # # form = ParameterForm(request.POST)
            # # context = {'form': form}
            # league = request.POST['region_choice']
            # split = request.POST['season_choice']
            # min_game_count = int(request.POST['min_games'])
            # top = crc.TopRadar(league, split, min_game_count)
            # top.create_radar()
            print('test')
        # elif 'test' in request.POST:
        #     league = request.POST['region_choice']
        #     split = request.POST['season_choice']
        #     min_game_count = int(request.POST['min_games'])
        #     top = crc.TopRadar(league, split, min_game_count)
        #     top.create_radar()
        #     print('hoge')
        form = ParameterForm(request.POST)
        context = {'form': form}
    return render(request, 'radar/index.html', context)
    # return render(request, 'radar/index.html')