from rrt_star_planning_tools import *


class RRTStar():
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 xrandArea,
                 yrandArea,
                 beta_opts,
                 search_until_max_iter=False,
                 connect_circle_dist=50.0,
                 velocity=4.0,
                 expand_dis=2.0,
                 power=4,
                 goal_sample_rate=15,
                 max_iter=500,
                 steps=4,
                 ):
        self.all_list = []
        self.tree_list = []
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_xrand = xrandArea[0]
        self.max_xrand = xrandArea[1]
        self.min_yrand = yrandArea[0]
        self.max_yrand = yrandArea[1]
        self.beta_opts = beta_opts
        self.search_until_max_iter = search_until_max_iter
        self.connect_circle_dist = connect_circle_dist
        self.velocity = velocity
        self.expand_dis = expand_dis
        self.power = power
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.steps = steps

    def cbf_rrt_star_planning(self, animation=True):
        self.tree_list = [self.start]
        self.all_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = get_random_node(self.goal_sample_rate, self.min_xrand, self.max_xrand, self.min_yrand, self.max_yrand, self.end)
            nearest_ind = get_nearest_node_index(self.tree_list, rnd_node)
            nearest_node = self.tree_list[nearest_ind]

            try:
                new_node, new_node_list = cbf_rrt_star_steer(nearest_node, rnd_node, self.beta_opts, self.steps, self.velocity)
            except ValueError:
                continue
            else:
                pass

            self.tree_list.append(new_node)

            for node in new_node_list:
                near_inds = find_near_nodes(node, self.all_list, self.connect_circle_dist, self.expand_dis)
                node_with_updated_parent = choose_parent(node, near_inds, self.all_list, self.beta_opts)
                rewire(node_with_updated_parent, near_inds, self.all_list, self.beta_opts)
                self.all_list.append(node_with_updated_parent)

            if animation:
                draw_graph(self.start,
                           self.end,
                           self.min_xrand,
                           self.max_xrand,
                           self.min_yrand,
                           self.max_yrand,
                           self.all_list,
                           self.obstacle_list,
                           self.beta_opts,
                           rnd_node)

            if ((not self.search_until_max_iter) and new_node):
                last_index =search_best_goal_node(self.end, self.all_list, self.expand_dis, self.beta_opts)
                if last_index is not None:
                    return generate_final_course(self.end, self.all_list, last_index)

        print("reached max iteration")
        last_index = search_best_goal_node(self.end, self.all_list, self.expand_dis, self.beta_opts)
        if last_index is not None:
            return generate_final_course(self.end, self.all_list, last_index)

        return None
