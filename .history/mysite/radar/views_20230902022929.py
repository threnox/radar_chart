from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect, FileResponse

from .views_sub import create_radar_chart as crc
from .forms import ParameterForm


def generator(request):

    # if request.method == 'POST':
    


    # league, split, min_game_count = 'LCK', 'Summer', 4
        # top = crc.TopRadar('LCK', 'Summer', 4)
    if request.method == "POST":
        
        if 'generate' in request.POST:
            # # form = ParameterForm(request.POST)
            # # context = {'form': form}
            league = request.POST['region_choice']
            split = request.POST['season_choice']
            min_game_count = int(request.POST['min_games'])
            top = crc.TopRadar(league, split, min_game_count)
            top.create_radar()
            print('test')
            # urls.pyで定義したname
            return HttpResponse("radar/test_image.png")
            # return redirect('index')
            # return FileResponse(open('radar/static', "rb"), as_attachment=False, filename='test_image.png')
        # elif 'test' in request.POST:
        #     league = request.POST['region_choice']
        #     split = request.POST['season_choice']
        #     min_game_count = int(request.POST['min_games'])
        #     top = crc.TopRadar(league, split, min_game_count)
        #     top.create_radar()
        #     print('hoge')
    # top = crc.TopRadar(league, split, min_game_count)
    # top.create_radar()
    form = ParameterForm(request.POST)
    context = {'form': form}
    return render(request, 'radar/index.html', context)
    # return render(request, 'radar/index.html')