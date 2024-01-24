from django.shortcuts import redirect, render
# from .views_sub import create_radar_chart as crc
from .views_sub import generate_radar_chart as grc
from .forms import ParameterForm

from django.conf import settings

def generator(request):

    if request.method == "POST":
        if 'generate' in request.POST:
            form = ParameterForm(request.POST)
            if form.is_valid():
                league = request.POST['region_choice']
                split = request.POST['season_choice']
                position = request.POST['position_choice']
                min_game_count = int(request.POST['min_games'])
                position = position.replace('jungle', 'jng').replace('support', 'sup')

                if settings.DEBUG: # localでは画像を生成して表示
                    chart = grc.GenerateRadar(league, split, position, min_game_count)
                    chart.generate_radar()
                    # if position == 'top':
                    #     top = crc.TopRadar(league, split, min_game_count)
                    #     top.create_radar()
                    # elif position == 'jungle':
                    #     jng = crc.JngRadar(league, split, min_game_count)
                    #     jng.create_radar()
                    # elif position == 'mid':
                    #     mid = crc.MidRadar(league, split, min_game_count)
                    #     mid.create_radar()
                    # elif position == 'bot':
                    #     bot = crc.BotRadar(league, split, min_game_count)
                    #     bot.create_radar()
                    # elif position == 'support':
                    #     sup = crc.SupRadar(league, split, min_game_count)
                    #     sup.create_radar()
                    request.session['radar_image'] = settings.MEDIA_URL + 'radar_image' + grc.RND + '.png'
                else: # 本番環境ではbase64でエンコードして表示
                    chart = grc.GenerateRadar(league, split, position, min_game_count)
                    radar_image = chart.generate_radar()
                    # if position == 'top':
                    #     top = crc.TopRadar(league, split, min_game_count)
                    #     radar_image = top.create_radar()
                    # elif position == 'jungle':
                    #     jng = crc.JngRadar(league, split, min_game_count)
                    #     radar_image = jng.create_radar()
                    # elif position == 'mid':
                    #     mid = crc.MidRadar(league, split, min_game_count)
                    #     radar_image = mid.create_radar()
                    # elif position == 'bot':
                    #     bot = crc.BotRadar(league, split, min_game_count)
                    #     radar_image = bot.create_radar()
                    # elif position == 'support':
                    #     sup = crc.SupRadar(league, split, min_game_count)
                    #     radar_image = sup.create_radar()
                    request.session['radar_image'] = 'data:image/png;base64,' + radar_image

                print(grc.IMG_PATH)
                return redirect('radar:display') # urls.pyで定義したname

    else:
        form = ParameterForm()
    context = {'title': 'LoL Esports RadarChart Generator',
                'form': form}
    return render(request, 'radar/index.html', context)

def display(request):
    return render(request, 'radar/display.html')