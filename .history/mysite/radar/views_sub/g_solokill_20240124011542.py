import numpy as np

# 必ず大文字小文字関係なくアルファベット順で集計すること！！！(旧版)
# LECはGroupStageを合算して集計

# Spring
# csv_file = '2023_LoL_esports0330.csv'

# Summer
csv_file = './radar/views_sub/2023_LoL_esports.csv'

# LCK Spring
LCK_Spring_top_solokill = {
    'Burdol': 16,
    'Canna': 10,
    'DnDn': 12,
    'Doran': 12,
    'DuDu': 9,
    'Kiin': 13,
    'Kingen': 7,
    'Morgan': 8,
    'Rascal': 16,
    'Zeus': 19,
}
LCK_Spring_mid_solokill = {
    'Bdd': 8,
    'BuLLDoG': 8,
    'Callme': 0,
    'Chovy': 10,
    'Clozer': 15,
    'FATE': 7,
    'FIESTA': 9,
    'Faker': 11,
    'Karis': 11,
    'ShowMaker': 12,
    'Zeka': 17,
}
LCK_Spring_jng_steal = {
    'Canyon': 4,
    'Clid': 7,
    'Croco': 2,
    'Cuzz': 6,
    'Juhan': 2,
    'Oner': 5,
    'Peanut': 2,
    'Sylvie': 2,
    'UmTi': 3,
    'Willer': 5,
    'YoungJae': 4,
}
# LCK_spring_top_solokill = np.array([16, 10, 12, 12, 9, 13, 7, 8, 16, 19])
# LCK_spring_mid_solokill = np.array([8, 8, 0, 10, 15, 11, 7, 9, 11, 12, 17])
# LCK_spring_jng_steal = np.array([4, 7, 2, 6, 2, 5, 2, 2, 3, 5, 4])

# LCK Summer
LCK_Summer_top_solokill = {
    'Burdol': 6,
    'Canna': 11,
    'Clear': 2,
    'DnDn': 6,
    'Doran': 19,
    'DuDu': 13,
    'Kiin': 5,
    'Kingen': 9,
    'Morgan': 5,
    'PerfecT': 0,
    'Rascal': 12,
    'Zeus': 10,
}
LCK_Summer_mid_solokill = {
    'Bdd': 5,
    'BuLLDoG': 4,
    'Chovy': 9,
    'Clozer': 4,
    'FATE': 3,
    'FIESTA': 5,
    'Faker': 6,
    'Feisty': 0,
    'Ivory': 1,
    'Karis': 3,
    'kyeahoo': 1,
    'Poby': 1,
    'Quad': 4,
    'ShowMaker': 4,
    'Zeka': 10,
}
LCK_Summer_jng_steal = {
    'Canyon': 4,
    'Clid': 4,
    'Croco': 2,
    'Cuzz': 4,
    'Grizzly': 3,
    'HamBak': 0,
    'Juhan': 0,
    'Oner': 5,
    'Peanut': 3,
    'Sylvie': 3,
    'UmTi': 1,
    'Willer': 2,
    'YoungJae': 5,
}
# LCK_summer_top_solokill = np.array([6, 11, 2, 6, 19, 13, 5, 9, 5, 0, 12, 10])
# LCK_summer_mid_solokill = np.array([5, 4, 9, 4, 6, 3, 0, 5, 1, 3, 1, 1, 4, 4, 10])
# LCK_summer_jng_steal = np.array([4, 4, 2, 4, 3, 0, 0, 5, 3, 3, 1, 2, 5])

# LEC Winter+Group #Group未出場
LEC_Winter_top_solokill = {
    'Adam': 6+3,
    'BrokenBlade': 3+3,
    'Chasy': 3+2,
    'Evi': 6+2,
    'Finn': 5+2,
    'Irrelevant': 7+3,
    'Odoamne': 3+0,#
    'Photon': 4+2,
    'Szygenda': 3+1,
    'Wunder': 0+0,#
}
LEC_Winter_mid_solokill = {
    'Caps': 1+2,
    'Dajor': 2+2,
    'Humanoid': 0+0,#
    'Larssen': 0+0,
    'Nisqy': 5+1,
    'Perkz': 1+1,
    'Ruby': 2+0,
    'Sertuss': 1+0,
    'Vetheo': 1+0,#
    'nuc': 1+1,
}
LEC_Winter_jng_steal = {
    '113': 0+0,
    'Bo': 0+2,
    'Elyoya': 0+0,
    'Jankos': 0+0,
    'Malrang': 2+1,
    'Markoon': 1+2,
    'Razork': 2+0,#
    'Sheo': 3+0,
    'Xerxe': 0+0,#
    'Yike': 2+0,
}
# LEC_winter_top_solokill = np.array([6, 3, 3, 6, 5, 7, 3, 4, 3, 0])
# LEC_winter_mid_solokill = np.array([1, 2, 0, 0, 5, 1, 1, 2, 1, 1])
# LEC_winter_jng_steal = np.array([0, 0, 0, 0, 2, 1, 2, 3, 0, 2])

# LEC Spring+Group #Group未出場
LEC_Spring_top_solokill = {
    'Adam': 5+1,
    'BrokenBlade': 5+2,
    'Chasy': 4+1,
    'Evi': 2+0,#
    'Finn': 2+4,
    'Irrelevant': 1+1,
    'Odoamne': 4+0,#
    'Oscarinin': 4+0,
    'Photon': 7+3,
    'Szygenda': 3+2,
}
LEC_Spring_mid_solokill = {
    'Abbedagge': 0+0,#
    'Caps': 8+1,
    'Humanoid': 1+1,
    'Larssen': 1+1,
    'LIDER': 2+1,
    'Nisqy': 2+1,
    'Perkz': 0+2,
    'Ruby': 3+0,#
    'Sertuss': 1+0,
    'Vetheo': 1+0,#
    'nuc': 0+2,
}
LEC_Spring_jng_steal = {
    '113': 0+2,
    'Bo': 2+0,
    'Elyoya': 1+0,
    'Jankos': 2+0,#
    'Malrang': 1+2,
    'Markoon': 0+0,
    'Razork': 2+0,
    'Sheo': 0+0,
    'Xerxe': 1+0,#
    'Yike': 2+0,
}
# LEC_spring_top_solokill = np.array([5, 5, 4, 2, 2, 1, 4, 4, 7, 3])
# LEC_spring_mid_solokill = np.array([0, 8, 1, 1, 2, 2, 0, 3, 1, 1, 0])
# LEC_spring_jng_steal = np.array([0, 2, 1, 2, 1, 0, 2, 0, 1, 2])

# LEC Summer+Group #Group未出場
LEC_Summer_top_solokill = {
    'Adam': 9+5,
    'Agresivoo': 0+0,#
    'BrokenBlade': 4+2,
    'Chasy': 4+1,
    'Evi': 1+1,
    'Finn': 5+0,#
    'Irrelevant': 2+4,
    'Odoamne': 1+1,
    'Oscarinin': 3+1,
    'Photon': 1+0,#
    'Szygenda': 6+0,
}
LEC_Summer_mid_solokill = {
    'Abbedagge': 2+0,
    'Caps': 2+1,
    'Humanoid': 2+1,
    'Larssen': 2+0,
    'LIDER': 3+0,#
    'Nisqy': 0+2,
    'Perkz': 1+0,#
    'Sertuss': 1+1,
    'Vetheo': 3+0,
    'nuc': 3+2,
}
LEC_Summer_jng_steal = {
    '113': 1+0,#
    'Bo': 0+0,#
    'Daglas': 1+0,#
    'Elyoya': 0+0,
    'Jankos': 3+0,
    'Malrang': 2+0,
    'Markoon': 0+1,
    'Peach': 1+0,
    'Razork': 4+0,
    'Sheo': 1+0,
    'Yike': 0+0,
}
# LEC_summer_top_solokill = np.array([9, 0, 4, 4, 1, 5, 2, 1, 3, 1, 6])
# LEC_summer_mid_solokill = np.array([2, 2, 2, 2, 3, 0, 3, 1, 1, 3])
# LEC_summer_jng_steal = np.array([1, 0, 1, 0, 3, 2, 0, 1, 4, 1, 0])

# LJL Spring
LJL_Spring_top_solokill = {
    'Ino': 10,
    'Kinatu': 10,
    'Nap': 4,
    'Paz': 8,
    'RayFarky': 11,
    'TaNa': 9,
    'Washidai': 5,
    'eguto': 1,
    'tol2': 12,
}
LJL_Spring_mid_solokill = {
    'Acee': 3,
    'Aria': 11,
    'DasheR': 8,
    'DICE': 13,
    'Eugeo': 3,
    'Jett': 21,
    'Megumiin': 2,
    'Recap': 12,
}
LJL_Spring_jng_steal = {
    'Blank': 3,
    'CaD': 0,
    'Cassin': 5,
    'EL': 2,
    'hachamecha': 1,
    'Hoglet': 1,
    'HRK': 2,
    'Once': 4,
    'Steal': 5,
}
# LJL_spring_top_solokill = np.array([1, 10, 10, 4, 8, 11, 9, 12, 5])
# LJL_spring_mid_solokill = np.array([3, 11, 8, 13, 3, 21, 2, 12])
# LJL_spring_jng_steal = np.array([3, 0, 5, 2, 1, 1, 2, 4, 5])

# LJL Summer
LJL_Summer_top_solokill = {
    'Ino': 1,
    'Kinatu': 5,
    'Nap': 2,
    'Paz': 2,
    'RayFarky': 5,
    'Ricky': 1,
    'TaNa': 7,
    'Washidai': 0,
    'Yutapon': 2,
    'eguto': 1,
    'tol2': 1,
}
LJL_Summer_mid_solokill = {
    'Acee': 1,
    'Aria': 8,
    'DasheR': 4,
    'DICE': 6,
    'Eugeo': 3,
    'Jett': 10,
    'Megumiin': 1,
    'Recap': 2,
}
LJL_Summer_jng_steal = {
    'Blank': 3,
    'Cassin': 3,
    'EL': 1,
    'hachamecha': 0,
    'Hoglet': 2,
    'HRK': 1,
    'Nesty': 2,
    'Once': 1,
    'Steal': 2,
}
# LJL_summer_top_solokill = np.array([1, 1, 5, 2, 2, 5, 1, 7, 1, 0, 2])
# LJL_summer_mid_solokill = np.array([1, 8, 4, 6, 3, 10, 1, 2])
# LJL_summer_jng_steal = np.array([3, 3, 1, 0, 2, 1, 2, 1, 2])
