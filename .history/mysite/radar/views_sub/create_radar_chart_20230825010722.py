import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from scipy import stats

import views_sub.radar_chart_original as rc
import views_sub.solokill_and_steal as sas


# importの下は2行開ける
np.set_printoptions(threshold=200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.options.display.max_seq_items = 200
pd.options.display.float_format = None

sns.set_theme()
sns.set_style('darkgrid', {'axes.edgecolor': '#bbbbd5'})

# global変数は大文字にする
CSV_FILE = sas.csv_file
MIN_GAME_COUNT = 3
DT_NOW = datetime.now()
# コメントは変数の上に書く
BASE_DF = pd.read_csv(CSV_FILE, parse_dates=['date'])
POSITION = ['top', 'jng', 'mid', 'bot', 'sup']
# # for use .str accessor
# POSITION_ORDER = {'top': str(0), 'jng': str(1), 'mid':str(2), 'bot': str(3), 'sup': str(4)}
BASE_STATS = ['playername', 'teamname', 'position', 'kills', 'deaths', 'assists', 'teamkills', 'firstbloodkill', 'firstbloodassist',
              'damagetochampions', 'dpm', 'wpm', 'wcpm', 'visionscore', 'vspm',
              'totalgold', 'earnedgold', 'earned gpm', 'total cs', 'cspm',
              'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

TOP_STATS = ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'earned gpm', 'dpm', 'KDA', 'KP', 'solokill']
JNG_STATS = ['golddiffat15', 'xpdiffat15', 'FB', 'earned gpm', 'dpm', 'KDA', 'KP', 'steal']
MID_STATS = ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'earned gpm', 'dpm', 'KDA', 'KP', 'solokill']
BOT_STATS = ['golddiffat15', 'xpdiffat15', 'csdiffat15', 'earned gpm', 'dpm', 'KDA', 'KP', 'DPG']
SUP_STATS = ['golddiffat15', 'xpdiffat15', 'FB', 'KDA', 'KP', 'vspm']

top_solokill_count = np.array([])
mid_solokill_count = np.array([])
jng_steal_count = np.array([])


class OriginalDf:

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        self.league = league
        self.split = split
        self.min_game_count = min_game_count
        self.playoffs = playoffs
        self.index = BASE_DF.index
        self.columns = BASE_DF.columns
        self.values = BASE_DF.values

        global top_solokill_count, mid_solokill_count, jng_steal_count
        if self.league == 'LCK':
            top_solokill_count = sas.LCK_top_solokill
            mid_solokill_count = sas.LCK_mid_solokill
            jng_steal_count = sas.LCK_jng_steal
        elif self.league == 'LEC':
            top_solokill_count = sas.LEC_top_solokill
            mid_solokill_count = sas.LEC_mid_solokill
            jng_steal_count = sas.LEC_jng_steal
        elif self.league == 'LJL':
            top_solokill_count = sas.LJL_top_solokill
            mid_solokill_count = sas.LJL_mid_solokill
            jng_steal_count = sas.LJL_jng_steal
        elif self.league == 'LPL':
            top_solokill_count = sas.LPL_top_solokill
            mid_solokill_count = sas.LPL_mid_solokill
            jng_steal_count = sas.LPL_jng_steal

    def extract(self): # クラス内メソッドの間は1行開ける
        df = BASE_DF.query(
            '(league == @self.league) & (split == @self.split) & (playoffs == @self.playoffs)')
        df = df.replace({'playername': {'Nesuty': 'Nesty'}})
        # 誤植箇所の修正
        return df

    def get_games_played(self):
        games_played = []
        for pos in POSITION:
            game_counts = self.extract().query('position == @pos')['playername'].value_counts()
            temp = pd.DataFrame(
                game_counts.values,
                columns=['games_played'],
                index=[game_counts.index, [pos for _ in range(len(game_counts))]])
            games_played.append(temp)
        df = pd.concat(games_played)
        df.index.names = ['playername', 'position']
        # レーンごとにプレイゲーム数を算出
        # 複数レーンをプレイしたプレイヤーに対応するためMultiIndexで作成
        return df


# globalクラス、関数の間は2行開ける
class DataProcessing(OriginalDf):

    def make_groups(self):
        # df = self.extract().loc[:, BASE_STATS]
        # df['order'] = df['position'].map(POSITION_ORDER)
        # df = df.groupby(by=['playername', 'teamname', 'position', 'order']).agg('sum', numeric_only=True)
        df = self.extract().loc[:, BASE_STATS].groupby(
            by=['playername', 'teamname', 'position']).agg('sum', numeric_only=True)
        df = df.join(self.get_games_played())
        df.sort_index(level=['teamname', 'playername'], inplace=True, key=lambda x: x.str.lower())
        # チーム名順にソート（チーム名が空欄のAverageを最後にするため）
        return df

    def create_solokill(self):
        global top_players, jng_players, mid_players, bot_players, sup_players
        top_players = self.make_groups().query(
            '(position == "top") & (teamname != "" "")'
            ).index.get_level_values('playername').sort_values(key=lambda x: x.str.lower())
        jng_players = self.make_groups().query(
            '(position == "jng") & (teamname != "" "")'
            ).index.get_level_values('playername').sort_values(key=lambda x: x.str.lower())
        mid_players = self.make_groups().query(
            '(position == "mid") & (teamname != "" "")'
            ).index.get_level_values('playername').sort_values(key=lambda x: x.str.lower())
        bot_players = self.make_groups().query(
            '(position == "bot") & (teamname != "" "")'
            ).index.get_level_values('playername').sort_values(key=lambda x: x.str.lower())
        sup_players = self.make_groups().query(
            '(position == "sup") & (teamname != "" "")'
            ).index.get_level_values('playername').sort_values(key=lambda x: x.str.lower())
        # 大文字小文字に対処しつつ名前順にソート

        try:
            top_solokill = pd.DataFrame(
                {'solokill': top_solokill_count, 'position': 'top'}, index=top_players)
            # position指定で複数レーンプレイヤーに対処
            mid_solokill = pd.DataFrame(
                {'solokill': mid_solokill_count, 'position': 'mid'}, index=mid_players)
        except ValueError:
            print('solokill count is not correct.')

        solokill = pd.concat([top_solokill, mid_solokill])
        solokill.set_index('position', append=True, inplace=True)
        # マルチインデックスとしてpositionを追加

        try:
            steal = pd.DataFrame(
                {'steal': jng_steal_count, 'position': 'jng'}, index=jng_players)
        except ValueError:
            print('steal count is not correct.')
        # プレイヤー数とカウント数の確認

        steal.set_index('position', append=True, inplace=True)
        return solokill, steal

    def get_total(self):
        df = self.make_groups().join(self.create_solokill()[0])
        df = df.join(self.create_solokill()[1])
        return df

    def get_average(self):
        df_total = self.get_total().swaplevel('position', 'teamname')
        for pos in POSITION:
            df_total.loc[('Average', '', pos), :] = df_total.query('position == @pos').agg('sum', numeric_only=True)
        df_average = df_total.div(df_total['games_played'], axis=0)
        df_average['games_played'] = df_total['games_played']
        # ゲーム数だけ平均値ではなく合計値に戻す
        df_average['KDA'] = (df_total['kills'] + df_total['assists']) / df_total['deaths']
        df_average['KP'] = (df_total['kills'] + df_total['assists'])*100 / df_total['teamkills']
        df_average['DPG'] = df_total['damagetochampions']*100 / df_total['totalgold']
        df_average['FB'] = (df_total['firstbloodkill'] + df_total['firstbloodassist'])*100 / df_total['games_played']
        # ゲーム数で割る前だと処理が面倒なのでdivの後でやる
        df_average.drop(
            df_average.index[df_average['games_played'] < self.min_game_count], axis=0, inplace=True)
        # 最少ゲーム数を設定
        return df_average


class TopDataFrame(DataProcessing):

    def get_top(self):
        df = self.get_average().xs('top', level='position').copy()
        # SettingWithCopyWarning対策
        df = df.loc[:, TOP_STATS]
        return df

    def make_top_labels(self):
        # プロットする数値の有効数字を調整
        df = self.get_top().round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'csdiffat15': 0,
            'earned gpm': 0,
            'dpm': 0,
            'KDA': 2,
            'KP': 1,
            'solokill': 2
            })
        df = df.astype({
            # 100.0→100 へ表示を変更するためにtypeを変更
            'golddiffat15': 'int64',
            'xpdiffat15': 'int64',
            'csdiffat15': 'int64',
            'earned gpm': 'int64',
            'dpm': 'int64'
            })
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'csdiffat15': 'CSD15',
            'earned gpm': 'EGPM',
            'dpm': 'DPM',
            'solokill': 'SoloKill'
            })

        for col in df.columns:
            def func_add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(func_add_colname)

        df['KP'] = df['KP'] + '%'
        df = df.reindex(columns=
            ['KDA', 'SoloKill', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15'])
        # columnsの順番を変更
        return df

    def standardize_top(self):
        # rounded前のdfを標準化して偏差値に変換
        # レーダーチャートのプロットは偏差値でおこなう
        df = stats.zscore(self.get_top(), axis=0)
        df = df.apply(lambda x: x*10 + 50)
        df = df.reindex(columns=
            ['KDA', 'solokill', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15'])
        # 順番をレーダーチャートに合わせる
        return df


class TopRadar(TopDataFrame):

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        super().__init__(league, split, min_game_count, playoffs)
        self.n = 8
        self.team = {}
        for key, value in self.make_top_labels().index:
            self.team[key] = value
        # プレイヤー名とチーム名の辞書を作成
        self.players = self.make_top_labels().reset_index('teamname').index
        # top_playersだとAverageがないので新しく作成
        temp_values = self.standardize_top().values.tolist()
        self.values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        self.labels = self.make_top_labels().values.tolist()
        # radar_chart.pyのdata形式に合わせる

    def create_radar(self, file_name=str(DT_NOW.strftime('%m')) + str(DT_NOW.strftime('%d'))):
        # axis_off, fig.suptitle, figsize, ax.set_xticklabels, fig.text, fig.savefigは随時調整
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(figsize=(16, 21), nrows=4, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(figsize=(16, 16), nrows=3, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        player_count = Counter(self.get_total().index.get_level_values('playername'))
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        # if len(set(self.get_total().index.get_level_values('playername')))\
        #     != len(self.get_total().index.get_level_values('playername')):
        #     print('Some players have played in multiple lanes.')

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

        # fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.values, self.labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team[player], size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.05), size=14, color='black')

        fig.suptitle(x=0.5, y=0.97, t=f'2023 {self.league} {self.split} TOP Stats',
                    horizontalalignment='center', color='black', weight='bold', size='30')

        if len(self.players) <= 11 or 13 <= len(self.players) < 16:
            fig.text(
                0.77, 0.05,
                'DPM: Damage Per Minute''\n'
                'DPG: Damage Per Gold''\n'
                'EGPM: Earned Gold Per Minute''\n'
                'VSPM: Vision Score Per Minute',
                horizontalalignment='left', color='black', size='16')

        # fig.savefig(f'2023{self.league}_{self.split}_top{file_name}.png', bbox_inches=None)
        fig.savefig('../static/radar/test_image.png', bbox_inches=None)

        # plt.show()


class JngDataFrame(DataProcessing):

    def get_jng(self):
        df = self.get_average().xs('jng', level='position').copy()
        df = df.loc[:, JNG_STATS]
        return df

    def make_jng_labels(self):
        df = self.get_jng().round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'FB': 1,
            'earned gpm': 0,
            'dpm': 0,
            'KDA': 2,
            'KP': 1,
            'steal': 2
            })
        df = df.astype({
            'golddiffat15': 'int64',
            'xpdiffat15': 'int64',
            'earned gpm': 'int64',
            'dpm': 'int64'
            })
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'earned gpm': 'EGPM',
            'dpm': 'DPM',
            'steal': 'Steal'
            })

        for col in df.columns:
            def func_add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(func_add_colname)

        df['KP'] = df['KP'] + '%'
        df['FB'] = df['FB'] + '%'
        df = df.reindex(columns=
            ['KDA', 'Steal', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'FB'])
        return df

    def standardize_jng(self):
        df = stats.zscore(self.get_jng(), axis=0)
        df = df.apply(lambda x: x*10 + 50)
        df = df.reindex(columns=
            ['KDA', 'steal', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'FB'])
        return df


class MidDataFrame(DataProcessing):

    def get_mid(self):
        df = self.get_average().xs('mid', level='position').copy()
        df = df.loc[:, MID_STATS]
        return df

    def make_mid_labels(self):
        df = self.get_mid().round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'csdiffat15': 0,
            'earned gpm': 0,
            'dpm': 0,
            'KDA': 2,
            'KP': 1,
            'solokill': 2
            })
        df = df.astype({
            'golddiffat15': 'int64',
            'xpdiffat15': 'int64',
            'csdiffat15': 'int64',
            'earned gpm': 'int64',
            'dpm': 'int64'
            })
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'csdiffat15': 'CSD15',
            'earned gpm': 'EGPM',
            'dpm': 'DPM',
            'solokill': 'SoloKill'
            })

        for col in df.columns:
            def func_add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(func_add_colname)

        df['KP'] = df['KP'] + '%'
        df = df.reindex(columns=
            ['KDA', 'SoloKill', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15'])
        return df

    def standardize_mid(self):
        df = stats.zscore(self.get_mid(), axis=0)
        df = df.apply(lambda x: x*10 + 50)
        df = df.reindex(columns=
            ['KDA', 'solokill', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15'])
        return df


class BotDataFrame(DataProcessing):
    def get_bot(self):
        df = self.get_average().xs('bot', level='position').copy()
        df = df.loc[:, BOT_STATS]
        return df

    def make_bot_labels(self):
        df = self.get_bot().round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'csdiffat15': 0,
            'earned gpm': 0,
            'dpm': 0,
            'KDA': 2,
            'KP': 1,
            'DPG': 0
            })
        df = df.astype({
            'golddiffat15': 'int64',
            'xpdiffat15': 'int64',
            'csdiffat15': 'int64',
            'earned gpm': 'int64',
            'dpm': 'int64',
            'DPG': 'int64'
            })
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'csdiffat15': 'CSD15',
            'earned gpm': 'EGPM',
            'dpm': 'DPM'
            })

        for col in df.columns:
            def func_add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(func_add_colname)

        df['KP'] = df['KP'] + '%'
        df['DPG'] = df['DPG'] + '%'
        df = df.reindex(columns=
            ['KDA', 'DPG', 'KP', 'DPM', 'EGPM', 'GD15', 'XPD15', 'CSD15'])
        return df

    def standardize_bot(self):
        df = stats.zscore(self.get_bot(), axis=0)
        df = df.apply(lambda x: x*10 + 50)
        df = df.reindex(columns=
            ['KDA', 'DPG', 'KP', 'dpm', 'earned gpm', 'golddiffat15', 'xpdiffat15', 'csdiffat15'])
        return df


class SupDataFrame(DataProcessing):
    def get_sup(self):
        df = self.get_average().xs('sup', level='position').copy()
        df = df.loc[:, SUP_STATS]
        return df

    def make_sup_labels(self):
        df = self.get_sup().round({
            'golddiffat15': 0,
            'xpdiffat15': 0,
            'FB': 1,
            'KDA': 2,
            'KP': 1,
            'vspm': 2
            })
        df = df.astype({
            'golddiffat15': 'int64',
            'xpdiffat15': 'int64'
            })
        df = df.rename(columns={
            'golddiffat15': 'GD15',
            'xpdiffat15': 'XPD15',
            'vspm': 'VSPM'
            })

        for col in df.columns:
            def func_add_colname(x): return col + '\n' + str(x)
            df[col] = df[col].map(func_add_colname)

        df['KP'] = df['KP'] + '%'
        df['FB'] = df['FB'] + '%'
        df = df.reindex(columns=
            ['KDA', 'VSPM', 'KP', 'GD15', 'XPD15', 'FB'])
        return df

    def standardize_sup(self):
        df = stats.zscore(self.get_sup(), axis=0)
        df = df.apply(lambda x: x*10 + 50)
        df = df.reindex(columns=
            ['KDA', 'vspm', 'KP', 'golddiffat15', 'xpdiffat15', 'FB'])
        return df


class JngRadar(JngDataFrame):

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        super().__init__(league, split, min_game_count, playoffs)
        self.n = 8
        self.team = {}
        for key, value in self.make_jng_labels().index:
            self.team[key] = value
        self.players = self.make_jng_labels().reset_index('teamname').index
        temp_values = self.standardize_jng().values.tolist()
        self.values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        self.labels = self.make_jng_labels().values.tolist()

    def create_radar(self, file_name=str(DT_NOW.strftime('%m')) + str(DT_NOW.strftime('%d'))):
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(figsize=(16, 21), nrows=4, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(figsize=(16, 16), nrows=3, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        player_count = Counter(self.get_total().index.get_level_values('playername'))
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        # if len(set(self.get_total().index.get_level_values('playername')))\
        #     != len(self.get_total().index.get_level_values('playername')):
        #     print('Some players have played in multiple lanes.')

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

        # fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.values, self.labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team[player], size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.05), size=14, color='black')

        fig.suptitle(x=0.5, y=0.97, t=f'2023 {self.league} {self.split} JNG Stats',
                    horizontalalignment='center', color='black', weight='bold', size='30')

        if len(self.players) <= 11 or 13 <= len(self.players) < 16:
            fig.text(0.77, 0.05,
                    'DPM: Damage Per Minute''\n'
                    'DPG: Damage Per Gold''\n'
                    'EGPM: Earned Gold Per Minute''\n'
                    'VSPM: Vision Score Per Minute',
                    horizontalalignment='left', color='black', size='16')

        fig.savefig(f'2023{self.league}_{self.split}_jng{file_name}.png', bbox_inches=None)

        # plt.show()


class MidRadar(MidDataFrame):

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        super().__init__(league, split, min_game_count, playoffs)
        self.n = 8
        self.team = {}
        for key, value in self.make_mid_labels().index:
            self.team[key] = value
        self.players = self.make_mid_labels().reset_index('teamname').index
        temp_values = self.standardize_mid().values.tolist()
        self.values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        self.labels = self.make_mid_labels().values.tolist()

    def create_radar(self, file_name=str(DT_NOW.strftime('%m')) + str(DT_NOW.strftime('%d'))):
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(figsize=(16, 21), nrows=4, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(figsize=(16, 16), nrows=3, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        player_count = Counter(self.get_total().index.get_level_values('playername'))
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        # if len(set(self.get_total().index.get_level_values('playername')))\
        #     != len(self.get_total().index.get_level_values('playername')):
        #     print('Some players have played in multiple lanes.')

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

        # fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.values, self.labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team[player], size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.05), size=14, color='black')

        fig.suptitle(x=0.5, y=0.97, t=f'2023 {self.league} {self.split} MID Stats',
                    horizontalalignment='center', color='black', weight='bold', size='30')

        if len(self.players) <= 11 or 13 <= len(self.players) < 16:
            fig.text(0.77, 0.05,
                    'DPM: Damage Per Minute''\n'
                    'DPG: Damage Per Gold''\n'
                    'EGPM: Earned Gold Per Minute''\n'
                    'VSPM: Vision Score Per Minute',
                    horizontalalignment='left', color='black', size='16')

        fig.savefig(f'2023{self.league}_{self.split}_mid{file_name}.png', bbox_inches=None)

        # plt.show()


class BotRadar(BotDataFrame):

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        super().__init__(league, split, min_game_count, playoffs)
        self.n = 8
        self.team = {}
        for key, value in self.make_bot_labels().index:
            self.team[key] = value
        self.players = self.make_bot_labels().reset_index('teamname').index
        temp_values = self.standardize_bot().values.tolist()
        self.values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        self.labels = self.make_bot_labels().values.tolist()

    def create_radar(self, file_name=str(DT_NOW.strftime('%m')) + str(DT_NOW.strftime('%d'))):
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(figsize=(16, 21), nrows=4, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(figsize=(16, 16), nrows=3, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        player_count = Counter(self.get_total().index.get_level_values('playername'))
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        # if len(set(self.get_total().index.get_level_values('playername')))\
        #     != len(self.get_total().index.get_level_values('playername')):
        #     print('Some players have played in multiple lanes.')

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

        # fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.values, self.labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team[player],
                    size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.05), size=14, color='black')

        fig.suptitle(x=0.5, y=0.97, t=f'2023 {self.league} {self.split} BOT Stats',
                    horizontalalignment='center', color='black', weight='bold', size='30')

        if len(self.players) <= 11 or 13 <= len(self.players) < 16:
            fig.text(0.77, 0.05,
                    'DPM: Damage Per Minute''\n'
                    'DPG: Damage Per Gold''\n'
                    'EGPM: Earned Gold Per Minute''\n'
                    'VSPM: Vision Score Per Minute',
                    horizontalalignment='left', color='black', size='16')

        fig.savefig(f'2023{self.league}_{self.split}_bot{file_name}.png', bbox_inches=None)

        # plt.show()


class SupRadar(SupDataFrame):

    def __init__(self, league, split, min_game_count=MIN_GAME_COUNT, playoffs=False):
        super().__init__(league, split, min_game_count, playoffs)
        self.n = 6
        self.team = {}
        for key, value in self.make_sup_labels().index:
            self.team[key] = value
        self.players = self.make_sup_labels().reset_index('teamname').index
        temp_values = self.standardize_sup().values.tolist()
        self.values = [[temp_values[i], temp_values[-1]] for i in range(len(temp_values))]
        self.labels = self.make_sup_labels().values.tolist()

    def create_radar(self, file_name=str(DT_NOW.strftime('%m')) + str(DT_NOW.strftime('%d'))):
        theta = rc.radar_factory(self.n, frame='polygon')

        if len(self.players) >= 13:
            fig, axs = plt.subplots(figsize=(16, 21), nrows=4, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.24, top=0.88, right=0.95, bottom=0, left=0.05)
        else:
            fig, axs = plt.subplots(figsize=(16, 16), nrows=3, ncols=4,
                                subplot_kw=dict(projection='radar'), facecolor='whitesmoke')
            fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        player_count = Counter(self.get_total().index.get_level_values('playername'))
        multi_player = [k for k, v in player_count.items() if v > 1]
        for p in multi_player:
            print(p + ' has played in multiple lanes.')
        # if len(set(self.get_total().index.get_level_values('playername')))\
        #     != len(self.get_total().index.get_level_values('playername')):
        #     print('Some players have played in multiple lanes.')

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

        # fig.subplots_adjust(wspace=0.6, hspace=0.06, top=0.88, right=0.95, bottom=0, left=0.05)

        colors = ['b', 'r']
        alphas = [0.95, 0.2]
        linewidths = [2.0, 0.5]

        for ax, player, case_data, label in zip(axs.flat, self.players, self.values, self.labels):
            ax.set_rgrids([20, 35, 50, 65, 80])
            ax.set_rlim([20, 80])
            ax.set_yticklabels([])
            ax.set_title(player, weight='bold', size=26, y=1.34, horizontalalignment='center')
            ax.text(0, 112, self.team[player],
                    size=18, horizontalalignment='center')

            for d, color, alpha, linewidth in zip(case_data, colors, alphas, linewidths):
                ax.plot(theta, d, color=color, alpha=alpha, linewidth=linewidth)
                ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
            ax.set_xticks(theta)
            ax.set_xticklabels(label, position=(0.0, -0.09), size=14, color='black')
            # Supだけちょっと広め

        fig.suptitle(x=0.5, y=0.97, t=f'2023 {self.league} {self.split} SUP Stats',
                    horizontalalignment='center', color='black', weight='bold', size='30')
        if len(self.players) <= 11 or 13 <= len(self.players) < 16:
            fig.text(0.77, 0.05,
                    'DPM: Damage Per Minute''\n'
                    'DPG: Damage Per Gold''\n'
                    'EGPM: Earned Gold Per Minute''\n'
                    'VSPM: Vision Score Per Minute',
                    horizontalalignment='left', color='black', size='16')
        fig.savefig(
            f'2023{self.league}_{self.split}_sup{file_name}.png', bbox_inches=None)

        # plt.show()
