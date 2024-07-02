from cbf_rrt import *
show_animation = True

points1 = [[5, 71], [18, 74], [21, 64], [7, 62]]
points2 = [[15, 40], [25, 47], [25, 38]]
points3 = [[49, 46], [51, 56], [60, 54], [57, 45]]
points4 = [[88, 32], [93, 35], [94, 30]]
points5 = [[50, 17], [56, 20], [57, 16], [52, 14]]

points = [[30, 20], [30, 50], [50, 60], [60, 20]]

# obs_list = [points1, points2, points3, points4, points5]
obs_list = [points]
sd = 4
# draw_poly(obs_list, sd)
beta_opts = multi_classify(obs_list, sd)
plan = RRT(
    start=[9, 8],
    goal=[74.5, 68],
    obstacle_list=obs_list,
    xrandArea=[0, 100],
    yrandArea=[0, 80],
    beta_opts=beta_opts,
    )
path = plan.cbf_rrt_planning(show_animation)

if path is None:
    print("Cannot find path")
else:
    print("found path!!")

if show_animation:
    draw_graph(
        plan.start,
        plan.end,
        plan.min_xrand,
        plan.max_xrand,
        plan.min_yrand,
        plan.max_yrand,
        plan.all_list,
        plan.obstacle_list,
        plan.beta_opts,
    )
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()