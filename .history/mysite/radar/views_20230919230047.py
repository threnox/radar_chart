from django.shortcuts import redirect, render
# from django.http import HttpResponse, HttpResponseRedirect, FileResponse, HttpResponseRedirect
import os
from os.path import dirname, abspath
import sys

# job_exex.pyから3つ上のディレクトリの絶対パスを取得し、sys.pathに登録する
parent_dir = dirname(dirname(dirname(abspath(__file__)))) # 追加
if parent_dir not in sys.path: # 追加
    sys.path.append(parent_dir) # 追加
from .views_sub import create_radar_chart as crc
from .forms import ParameterForm


def generator(request):

    if request.method == "POST":
        if 'generate' in request.POST:
            form = ParameterForm(request.POST)
            if form.is_valid():
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
                request.session['radar_image'] = '/static/radar/images/radar_image' + crc.rnd + '.png'
                # return redirect('index') # urls.pyで定義したname
                return redirect('radar:display') # urls.pyで定義したname

    else:
        form = ParameterForm()
    context = {'title': 'LoL Esports RadarChart Generator',
                'form': form}
    return render(request, 'radar/index.html', context)

def display(request):
    return render(request, 'radar/display.html')