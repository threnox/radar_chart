from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect, FileResponse, HttpResponseRedirect

from .views_sub import create_radar_chart as crc
from .forms import ParameterForm


def generator(request):

    # if request.method == 'POST':
    


    # league, split, min_game_count = 'LCK', 'Summer', 4
        # top = crc.TopRadar('LCK', 'Summer', 4)
    if request.method == "POST":
        
        if 'generate' in request.POST:
            form = ParameterForm(request.POST)
            if form.is_valid():
                
                # # form = ParameterForm(request.POST)
                # # context = {'form': form}
                league = request.POST['region_choice']
                split = request.POST['season_choice']
                position = request.POST['position_choice']
                min_game_count = int(request.POST['min_games'])
                
                if position == 'top':
                    top = crc.TopRadar(league, split, min_game_count)
                    top.create_radar()
                elif position == 'jungle':
                    jng = crc.JngRadar(league, split, min_game_count)
                    jng.create_radar()
                elif position == 'mid':
                    mid = crc.MidRadar(league, split, min_game_count)
                    mid.create_radar()
                elif position == 'bot':
                    bot = crc.BotRadar(league, split, min_game_count)
                    bot.create_radar()
                elif position == 'support':
                    sup = crc.SupRadar(league, split, min_game_count)
                    sup.create_radar()
                # print('test')
                
                # return redirect('index') # urls.pyで定義したname
                return redirect('radar:display') # urls.pyで定義したname
            # else:
            #     return redirect('radar:index')
            # return HttpResponse('<img src="/static/radar/test_image.png">')
            # return HttpResponseRedirect('http://127.0.0.1:8000/radar/')
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
    else:
        form = ParameterForm()
    context = {'title': 'LoL Esports RadarChart Generator',
                'form': form}
    return render(request, 'radar/index.html', context)
    # return render(request, 'radar/index.html')
    
def display(request):
    return render(request, 'radar/display.html')