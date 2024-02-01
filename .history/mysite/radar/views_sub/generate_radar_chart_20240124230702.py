import base64
import glob
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # バックエンド指定
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
from collections import Counter
from datetime import datetime
from scipy import stats

from . import radar_chart_original as rc
from . import g_solokill as sk

from django.conf import settings

sns.set_theme()
sns.set_style('darkgrid', {'axes.edgecolor': '#d5bbbb'})

# global variable
CSV_FILE = sk.csv_file
MIN_GAME_COUNT = 4
DT_NOW = datetime.now()

IMG_DIR = settings.MEDIA_ROOT
# ブラウザキャッシュ対策に乱数を追加
RND = str(random.randint(0, 100000))
IMG_PATH = IMG_DIR + '/radar_image' + RND + '.png'
BASE_DF = pd.read_csv(CSV_FILE, parse_dates=['date'])
POSITION = ['top', 'jng', 'mid', 'bot', 'sup']

BASE_STATS = [
    'playername', 'teamname', 'position', 'kills', 'deaths', 'assists', 'teamkills', 'firstbloodkill', 'firstbloodassist',
    'damagetochampions', 'dpm', 'wpm', 'wcpm', 'visionscore', 'vspm',
    'totalgold', 'earnedgold', 'earned gpm', 'total cs', 'cspm',
    'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']
RADAR_STATS = [
    'KDA', 'solokill', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'steal', 'FB', 'DPG', 'vspm', 'games_played']


# Order of graphing
TOP_STATS = ['KDA', 'solokill', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15']
JNG_STATS = ['KDA', 'steal', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'FB']
MID_STATS = ['KDA', 'solokill', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15']
BOT_STATS = ['KDA', 'DPG', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15']
SUP_STATS = ['KDA', 'vspm', 'KP', 'golddiffat15', 'xpdiffat15', 'FB']

TOP_STATS_LABEL = ['KDA', 'SoloKill', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15']
JNG_STATS_LABEL = ['KDA', 'Steal', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'FB']
MID_STATS_LABEL = ['KDA', 'SoloKill', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15']
BOT_STATS_LABEL = ['KDA', 'DPG', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15']
SUP_STATS_LABEL = ['KDA', 'VSPM', 'KP', 'GD15', 'XPD15', 'FB']

TOP_SOLOKILL_COUNT = {}
MID_SOLOKILL_COUNT = {}
JNG_STEAL_COUNT = {}

# data:image/svg+xml;base64, 描画用
def output():
	buffer = BytesIO() # binary I/O (画像や音声データ向け)
	plt.savefig(buffer, format='png', bbox_inches=None)
	buffer.seek(0) # ストリーム先頭のoffset byteに変更
	img = buffer.getvalue()
	graph = base64.b64encode(img) # base64でエンコード
	graph = graph.decode('utf-8') # decodeして文字列から画像に変換
	buffer.close()
	return graph


class Preprocess:

    def __init__(self, league, split):
        self.league = league
        self.split = split

        global TOP_SOLOKILL_COUNT, MID_SOLOKILL_COUNT, JNG_STEAL_COUNT
        if self.league == 'LCK':
            if self.split == 'Spring':
                TOP_SOLOKILL_COUNT = sk.LCK_Spring_top_solokill
                MID_SOLOKILL_COUNT = sk.LCK_Spring_mid_solokill
                JNG_STEAL_COUNT = sk.LCK_Spring_jng_steal
            elif self.split == 'Summer':
                TOP_SOLOKILL_COUNT = sk.LCK_Summer_top_solokill
                MID_SOLOKILL_COUNT = sk.LCK_Summer_mid_solokill
                JNG_STEAL_COUNT = sk.LCK_Summer_jng_steal
        elif self.league == 'LEC':
            if self.split == 'Winter':
                TOP_SOLOKILL_COUNT = sk.LEC_Winter_top_solokill
                MID_SOLOKILL_COUNT = sk.LEC_Winter_mid_solokill
                JNG_STEAL_COUNT = sk.LEC_Winter_jng_steal
            elif self.split == 'Spring':
                TOP_SOLOKILL_COUNT = sk.LEC_Spring_top_solokill
                MID_SOLOKILL_COUNT = sk.LEC_Spring_mid_solokill
                JNG_STEAL_COUNT = sk.LEC_Spring_jng_steal
            elif self.split == 'Summer':
                TOP_SOLOKILL_COUNT = sk.LEC_Summer_top_solokill
                MID_SOLOKILL_COUNT = sk.LEC_Summer_mid_solokill
                JNG_STEAL_COUNT = sk.LEC_Summer_jng_steal
        elif self.league == 'LJL':
            if self.split == 'Spring':
                TOP_SOLOKILL_COUNT = sk.LJL_Spring_top_solokill
                MID_SOLOKILL_COUNT = sk.LJL_Spring_mid_solokill
                JNG_STEAL_COUNT = sk.LJL_Spring_jng_steal
            elif self.split == 'Summer':
                TOP_SOLOKILL_COUNT = sk.LJL_Summer_top_solokill
                MID_SOLOKILL_COUNT = sk.LJL_Summer_mid_solokill
                JNG_STEAL_COUNT = sk.LJL_Summer_jng_steal

    def slice_data(self):
        df = BASE_DF.query('(league == @self.league) & (split == @self.split) & (playoffs == 0)')
        df = df.loc[:, BASE_STATS]
        # fix misprint
        df = df.replace({'playername': {'Nesuty': 'Nesty'}})

        return df


class DataProcessing(Preprocess):

    def __init__(self, league, split):
        super().__init__(league, split)

    def calc_games_played(self):
        """Calculate the number of games played per lane.
        Create with MultiIndex to apply players who played multiple lanes.
        """
        games_played = []
        for pos in POSITION:
            game_counts = self.slice_data().query('position == @pos')['playername'].value_counts()
            temp = pd.DataFrame(
                game_counts.values,
                columns=['games_played'],
                index=[game_counts.index, [pos for _ in range(len(game_counts))]])
            games_played.append(temp)
        df = pd.concat(games_played)
        # use MultiIndex to deal with players who played multiple lanes
        df.index.names = ['playername', 'position']
        return df

    def create_solokill(self):

        try:
            top_solokill = pd.DataFrame.from_dict(TOP_SOLOKILL_COUNT, orient='index', columns=['solokill'])
            # position指定で複数レーンプレイヤーに対処
            top_solokill['position'] = 'top'
            top_solokill.index.set_names('playername', inplace=True)
            mid_solokill = pd.DataFrame.from_dict(MID_SOLOKILL_COUNT, orient='index', columns=['solokill'])
            mid_solokill.index.set_names('playername', inplace=True)
            mid_solokill['position'] = 'mid'
        except ValueError:
            print('solokill count is not correct.')

        solokill = pd.concat([top_solokill, mid_solokill])
        # マルチインデックスとしてpositionを追加
        solokill.set_index('position', append=True, inplace=True)

        # プレイヤー数とカウント数の確認
        try:
            steal = pd.DataFrame.from_dict(JNG_STEAL_COUNT, orient='index', columns=['steal'])
            steal['position'] = 'jng'
            steal.index.set_names('playername', inplace=True)
        except ValueError:
            print('steal count is not correct.')
        steal.set_index('position', append=True, inplace=True)

        solokill_and_steal = pd.concat([solokill, steal])

        return solokill_and_steal

    def calc_agg_sum(self):
        df =  self.slice_data().groupby(by=['playername', 'teamname', 'position']).agg('sum', numeric_only=True)
        df = df.join(self.calc_games_played())
        df = df.join(self.create_solokill())
        # チーム名順にソート（チーム名が空欄のAverageを最後にするため）
        df.sort_index(level=['teamname', 'playername'], inplace=True, key=lambda x: x.str.lower())
        return df

    def calc_average(self):
        df_total = self.calc_agg_sum().swaplevel('position', 'teamname')
        for pos in POSITION:
            df_total.loc[('Average', '', pos), :] = df_total.query('position == @pos').agg('sum', numeric_only=True)
        df_average = df_total.div(df_total['games_played'], axis=0)
        # ゲーム数だけ平均値ではなく合計値に戻す
        df_average['games_played'] = df_total['games_played']
        df_average['KDA'] = (df_total['kills'] + df_total['assists']) / df_total['deaths']
        df_average['KP'] = (df_total['kills'] + df_total['assists'])*100 / df_total['teamkills']
        df_average['DPG'] = df_total['damagetochampions']*100 / df_total['totalgold']
        df_average['FB'] = (df_total['firstbloodkill'] + df_total['firstbloodassist'])*100 / df_total['games_played']
        # ゲーム数で割る前だと処理が面倒なのでdivの後でスライス
        df_average = df_average.loc[:, RADAR_STATS]

        return df_average

    def create_csv(self):
        self.calc_average().to_csv(f'{self.league}_{self.split}_processing.csv')
        player_count = Counter(self.calc_agg_sum().index.get_level_values('playername'))
        # min_game＿countでdropする前のdfを参照する
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        print('created csv file')


class RadarProcessing:

    # 軽量化のためcsvを分離
    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT):
        self.league = league
        self.split = split
        self.min_game_count = min_game_count
        self.df_base = pd.read_csv(
            f'./radar/views_sub/{self.league}_{self.split}_processing.csv', index_col=['playername', 'teamname', 'position'])

        if self.min_game_count > 8:
            raise ValueError('min_game_count must be less than 8')

    def get_df(self):
        df = self.df_base
        df = df.rename(index={np.nan: ''})
        # 最少ゲーム数を適用
        df = df[df['games_played'] >= self.min_game_count]
        return df

    def create_labels(self):
        """Adjust the significant figures of plotting number
        """
        df = self.get_df()

        # プロットする数値の有効数字を調整
        df = df.round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'csdiffat15': 0,
            'FB': 1,
            'earned gpm': 0,
            'dpm': 0,
            'KDA': 2,
            'KP': 1,
            'solokill': 2,
            'steal': 2,
            'DPG': 0,
            'vspm': 2})
        # '100.0'→'100' へ表示を変更するためにtypeを変更
        df = df.astype({
            'golddiffat15': 'int16',
            'xpdiffat15': 'int16',
            'csdiffat15': 'int16',
            'earned gpm': 'int16',
            'dpm': 'int16',
            'DPG': 'int16'})
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'csdiffat15': 'CSD15',
            'earned gpm': 'EGPM',
            'dpm': 'DPM',
            'solokill': 'SoloKill',
            'steal': 'Steal',
            'vspm': 'VSPM'})

        for col in df.columns:
            def add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(add_colname)

        df['KP'] = df['KP'] + '%'
        df['FB'] = df['FB'] + '%'
        df['DPG'] = df['DPG'] + '%'

        return df


class GenerateRadar(RadarProcessing):

    def __init__(self, league, split, position, min_game_count=MIN_GAME_COUNT):
        super().__init__(league, split, min_game_count)

        self.position = position
        self.games_played = self.df_base['games_played']
        self.team_dict = {}
        self.n = 6 if self.position == 'sup' else 8

        if self.position == 'top':
            self.stats, self.stats_label = TOP_STATS, TOP_STATS_LABEL
        elif self.position == 'jng':
            self.stats, self.stats_label = JNG_STATS, JNG_STATS_LABEL
        elif self.position == 'mid':
            self.stats, self.stats_label = MID_STATS, MID_STATS_LABEL
        elif self.position == 'bot':
            self.stats, self.stats_label = BOT_STATS, BOT_STATS_LABEL
        elif self.position == 'sup':
            self.stats, self.stats_label = SUP_STATS, SUP_STATS_LABEL
        # rounded前のdfを偏差値に変換後、レーダーチャートにプロット
        self.df_avg = self.get_df().xs(self.position, level='position').loc[:, self.stats]
        self.df_std = stats.zscore(self.df_avg, axis=0).apply(lambda x: x*10 + 50)
        self.df_label = self.create_labels().xs(self.position, level='position').loc[:, self.stats_label]

        if self.df_avg.isna().sum().sum() !=0:
            raise ValueError('DataFrame contains NaN. Probably solokill.py is not correct.')

        # プレイヤー名とチーム名の辞書を作成
        for key, value in self.df_label.index:
            self.team_dict[key] = value

        self.players = self.df_label.reset_index('teamname').index
        temp_values = self.df_std.values.tolist()
        # radar_chart.pyのdata形式に合わせる
        # plotはrounded前
        self.plot_values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        # ラベルはrounded後
        self.plot_labels = self.df_label.values.tolist()

        self.text = 'DPM: Damage Per Minute'\
            '\n''DPG: Damage Per Gold'\
            '\n''EGPM: Earned Gold Per Minute'\
            '\n''VSPM: Vision Score Per Minute'\
            '\n'f'min game count = {self.min_game_count}'

    def generate_radar(self, file_name=str(DT_NOW.strftime('%m%d'))):
        # axis_off, fig.suptitle, figsize, ax.set_xticklabels, fig.text, fig.savefigは随時調整
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(
                figsize=(16, 21), nrows=4, ncols=4, subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(
                figsize=(16, 16), nrows=3, ncols=4, subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        if len(self.players) >= 17:
            print('Too many players to show')
        elif len(self.players) == 15:
            axs[3, 3].axis('off')
        elif len(self.players) == 14:
            axs[3, 3].axis('off')
            axs[3, 2].axis('off')
        elif len(self.players) == 13:
            axs[3, 3].axis('off')
            axs[3, 2].axis('off')
            axs[3, 1].axis('off')
        elif len(self.players) == 11:
            axs[2, 3].axis('off')
        elif len(self.players) == 10:
            axs[2, 3].axis('off')
            axs[2, 2].axis('off')
        elif len(self.players) == 9:
            axs[2, 3].axis('off')
            axs[2, 2].axis('off')
            axs[2, 1].axis('off')

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.plot_values, self.plot_labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team_dict[player], size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.05), size=14, color='black')
            if self.position == 'sup':
                ax.set_xticklabels(label, position=(0.0, -0.09), size=14, color='black')

        fig.suptitle(
            x=0.5, y=0.97, t=f'2023 {self.league} {self.split} {self.position.upper()} Stats',
            horizontalalignment='center', color='black', weight='bold', size='30')

        if len(self.players) <= 11:
            fig.text(0.77, 0.05, self.text, horizontalalignment='left', color='black', size='16')
        elif 13 <= len(self.players) < 16:
            fig.text(0.77, 0.03, self.text, horizontalalignment='left', color='black', size='16')

        if settings.DEBUG:
        # ローカル環境では古い画像を削除、新しい画像を出力して保存
        # 本番環境ではbase64でエンコードして表示
            for p in glob.glob(f'{IMG_DIR}/radar_image*.png'):
                if os.path.isfile(p):
                    os.remove(p)
            fig.savefig(IMG_PATH, bbox_inches=None)
        else:
            graph = output()
            return graph

        # fig.savefig(f'{DT_NOW.strftime("%Y")}{self.league}_{self.split}_{self.position}{file_name}.png', bbox_inches=None)
        # plt.show()

# Preprocess('LCK', 'Summer').slice_data()

# DataProcessing('LCK', 'Summer').calc_games_played()

# DataProcessing('LCK', 'Summer').create_solokill()

# DataProcessing('LCK', 'Summer').calc_agg_sum()

# DataProcessing('LCK', 'Summer').calc_average()

# DataProcessing('LEC', 'Winter').create_csv()
# DataProcessing('LEC', 'Spring').create_csv()
# DataProcessing('LEC', 'Summer').create_csv()
# DataProcessing('LCK', 'Spring').create_csv()
# DataProcessing('LCK', 'Summer').create_csv()
# DataProcessing('LJL', 'Spring').create_csv()
# DataProcessing('LJL', 'Summer').create_csv()
