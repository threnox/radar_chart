import numpy as np

# 必ず大文字小文字関係なくアルファベット順で集計すること！！！

csv_file = 'radar/views_sub/2023_LoL_esports.csv'

# LEC Winter
LEC_top_solokill_winter = np.array([6, 3, 3, 6, 5, 7, 3, 4, 3, 0])
LEC_mid_solokill_winter = np.array([1, 2, 0, 0, 5, 1, 1, 2, 1, 1])
LEC_jng_steal_winter = np.array([0, 0, 0, 0, 2, 1, 2, 3, 0, 2])

# LCK Spring
LCK_top_solokill_spring = np.array([16, 10, 12, 12, 9, 13, 7, 8, 16, 19])
LCK_mid_solokill_spring = np.array([8, 8, 0, 10, 15, 11, 7, 9, 11, 12, 17])
LCK_jng_steal_spring = np.array([4, 7, 2, 6, 2, 5, 2, 2, 3, 5, 4])

# LEC Spring
LEC_top_solokill_spring = np.array([5, 5, 4, 2, 2, 1, 4, 4, 7, 3])
LEC_mid_solokill_spring = np.array([0, 8, 1, 1, 2, 2, 0, 3, 1, 1, 0])
LEC_jng_steal_spring = np.array([0, 2, 1, 2, 1, 0, 2, 0, 1, 2])

# LJL Spring
LJL_top_solokill_spring = np.array([1, 10, 10, 4, 8, 11, 9, 12, 5])
LJL_mid_solokill_spring = np.array([3, 11, 8, 13, 3, 21, 2, 12])
LJL_jng_steal_spring = np.array([3, 0, 5, 2, 1, 1, 2, 4, 5])

# LCK Summer
LCK_top_solokill_summer = np.array([6, 11, 2, 6, 19, 13, 5, 9, 5, 0, 12, 10])
LCK_mid_solokill_summer = np.array([5, 4, 9, 4, 6, 3, 0, 5, 1, 3, 1, 1, 4, 4, 10])
LCK_jng_steal_summer = np.array([4, 4, 2, 4, 3, 0, 0, 5, 3, 3, 1, 2, 5])

# LEC Summer
LEC_top_solokill_summer = np.array([9, 0, 4, 4, 1, 5, 2, 1, 3, 1, 6])
LEC_mid_solokill_summer = np.array([2, 2, 2, 2, 3, 0, 3, 1, 1, 3])
LEC_jng_steal_summer = np.array([1, 0, 1, 0, 3, 2, 0, 1, 4, 1, 0])

# LJL Summer
LJL_top_solokill_summer = np.array([1, 1, 5, 2, 2, 5, 1, 7, 1, 0, 2])
LJL_mid_solokill_summer = np.array([1, 8, 4, 6, 3, 10, 1, 2])
LJL_jng_steal_summer = np.array([3, 3, 1, 0, 2, 1, 2, 1, 2])
