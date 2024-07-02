from rrt_planning_tools import *

class RRT:
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 xrandArea,
                 yrandArea,
                 beta_opts,
                 velocity=4.0,
                 expand_dis=2.0,
                 power=4,
                 goal_sample_rate=15,
                 max_iter=5000,
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
        self.velocity = velocity
        self.expand_dis = expand_dis
        self.power = power
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.steps = steps

    def cbf_rrt_planning(self, animation=True):
        self.tree_list = [self.start]
        self.all_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = get_random_node(self.goal_sample_rate, self.min_xrand, self.max_xrand, self.min_yrand, self.max_yrand, self.end)
            nearest_ind = get_nearest_node_index(self.tree_list, rnd_node)
            nearest_node = self.tree_list[nearest_ind]

            try:
                new_node, new_node_list = cbf_rrt_steer(nearest_node, rnd_node, self.beta_opts, self.steps, self.velocity)
            except ValueError:
                continue
            else:
                pass

            self.tree_list.append(new_node)
            for node in new_node_list:
                self.all_list.append(node)

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

            if calc_dist_to_goal(self.tree_list[-1], self.end) <= self.expand_dis:
                final_node = get_final_node(self.tree_list[-1], self.end)
                if check_collision(final_node, self.beta_opts):
                    return generate_final_course(self.end, self.all_list, len(self.all_list) - 1)

        return None
