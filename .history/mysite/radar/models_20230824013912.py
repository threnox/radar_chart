from django.db import models
import modelset.create_radar_chart as crc


# POSITION_ORDER = {'top': str(0), 'jng': str(1), 'mid':str(2), 'bot': str(3), 'sup': str(4)}
# POSITION = ['top', 'jng', 'mid', 'bot', 'sup']
# BASE_STATS = ['playername', 'teamname', 'position', 'kills', 'deaths', 'assists', 'teamkills', 'firstbloodkill', 'firstbloodassist',
#               'damagetochampions', 'dpm', 'wpm', 'wcpm', 'visionscore', 'vspm',
#               'totalgold', 'earnedgold', 'earned gpm', 'total cs', 'cspm',
#               'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

top = crc.TopRadar('LCK', 'Summer', 4)
# jng = crc.JngRadar('LCK', 'Summer', 4)
# mid = crc.MidRadar('LCK', 'Summer', 4)
# bot = crc.BotRadar('LCK', 'Summer', 4)
# sup = crc.SupRadar('LCK', 'Summer', 4)

# for pos in [top, jng, mid, bot, sup]:
#     pos.create_radar()
class CreateRadar(models.Model):
    top = crc.TopRadar('LCK', 'Summer', 4)
    top.create_radar()