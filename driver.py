from log_reg import *

points1 = [[5, 71], [18, 74], [21, 64], [7, 62]]
points2 = [[15, 40], [25, 47], [25, 38]]
points3 = [[49, 46], [51, 56], [60, 54], [57, 45]]
points4 = [[88, 32], [93, 35], [94, 30]]
points5 = [[50, 17], [56, 20], [57, 16], [52, 14]]

obs_list = [points1, points2, points3, points4, points5]
sd = 4
# draw_poly(obs_list, sd)
beta_opts = multi_classify(obs_list, sd)
