import math
import random

from rrt_planning_tools import (
    _EPS,
    _TOTAL_STEER_TIME,
    barrier_function,
    barrier_function_derivative,
    barrier_function_second_derivative,
    calc_dist_to_goal,
    calc_distance_and_angle,
    check_collision,
    draw_graph,
    generate_final_course,
    _solve_cbf_qp,
)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0


def cbf_rrt_star_steer(nearest_node, rnd_node, beta_opts, steps, v):
    num_steps = int(steps)
    if num_steps <= 0:
        raise ValueError("steps must be positive")
    if v <= 0:
        raise ValueError("velocity must be positive")

    dt = _TOTAL_STEER_TIME / num_steps
    step_distance = dt * v
    distance_to_sample, theta = calc_distance_and_angle(nearest_node, rnd_node)
    remaining_distance = min(distance_to_sample, step_distance * num_steps)
    if remaining_distance <= _EPS:
        raise ValueError("sample is already at the nearest node")

    new_node_list = []
    from_node = nearest_node
    new_node = None

    for _ in range(num_steps):
        if remaining_distance <= _EPS:
            break

        current_step = min(step_distance, remaining_distance)
        preview_node = get_new_node(from_node, theta, current_step)
        w = _solve_cbf_qp(beta_opts, preview_node.x, preview_node.y, theta, v)
        theta += dt * w

        new_node = get_new_node(from_node, theta, current_step)
        new_node_list.append(new_node)
        from_node = new_node
        remaining_distance -= current_step

    if new_node is None:
        raise ValueError("failed to generate a new node")

    return new_node, new_node_list


def find_near_nodes(new_node, all_list, connect_circle_dist, expand_dis):
    nnode = len(all_list) + 1
    r = connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
    r = min(r, expand_dis)
    dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                 for node in all_list]
    near_inds = [i for i, distance in enumerate(dist_list) if distance <= r ** 2]
    return near_inds


def choose_parent(new_node, near_inds, all_list, beta_opts):
    if len(near_inds) == 0:
        return new_node

    original_parent = new_node.parent
    best_parent = None
    best_cost = float("inf")

    for i in near_inds:
        candidate_parent = all_list[i]
        if candidate_parent is new_node:
            continue
        new_node.parent = candidate_parent
        if check_collision(new_node, beta_opts):
            d, _ = calc_distance_and_angle(candidate_parent, new_node)
            candidate_cost = d + candidate_parent.cost
            if candidate_cost < best_cost:
                best_parent = candidate_parent
                best_cost = candidate_cost

    new_node.parent = original_parent
    if best_parent is None:
        return new_node

    _set_parent(new_node, best_parent)
    new_node.cost = best_cost
    return new_node


def rewire(new_node, near_inds, all_list, beta_opts):
    for i in near_inds:
        near_node = all_list[i]
        if near_node is new_node or near_node.parent is None:
            continue
        if _is_ancestor(near_node, new_node):
            continue

        d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
        new_cost = new_node.cost + d
        if near_node.cost <= new_cost:
            continue

        old_parent = near_node.parent
        old_path_x = near_node.path_x[:]
        old_path_y = near_node.path_y[:]
        _set_parent(near_node, new_node)
        if check_collision(near_node, beta_opts):
            near_node.cost = new_cost
            _propagate_cost_to_leaves(near_node, all_list)
        else:
            near_node.parent = old_parent
            near_node.path_x = old_path_x
            near_node.path_y = old_path_y


def search_best_goal_node(end, all_list, expand_dis, beta_opts):
    dist_to_goal_list = [
        calc_dist_to_goal(n, end) for n in all_list
    ]
    goal_inds = [
        i for i, distance in enumerate(dist_to_goal_list)
        if distance <= expand_dis
    ]

    safe_goal_inds = []
    for goal_ind in goal_inds:
        goal_node = Node(end.x, end.y)
        goal_node.parent = all_list[goal_ind]
        if check_collision(goal_node, beta_opts):
            safe_goal_inds.append(goal_ind)

    if not safe_goal_inds:
        return None

    safe_goal_costs = [all_list[i].cost +
                       calc_dist_to_goal(all_list[i], end)
                       for i in safe_goal_inds]

    min_cost = min(safe_goal_costs)
    for i, cost in zip(safe_goal_inds, safe_goal_costs):
        if cost == min_cost:
            return i

    return None


def get_random_node(goal_sample_rate, min_xrand, max_xrand, min_yrand, max_yrand, end):
    if random.randint(0, 100) > goal_sample_rate:
        rnd = Node(
            random.uniform(min_xrand, max_xrand),
            random.uniform(min_yrand, max_yrand))
    else:
        rnd = Node(end.x, end.y)
    return rnd


def get_nearest_node_index(tree_list, rnd_node):
    dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in tree_list]
    min_ind = dlist.index(min(dlist))

    return min_ind


def get_new_node(from_node, theta, distance):
    if distance < -_EPS:
        raise ValueError("distance must be non-negative")

    new_node = Node(from_node.x, from_node.y)
    new_node.path_x = [new_node.x]
    new_node.path_y = [new_node.y]
    distance = max(0.0, distance)
    new_node.x += distance * math.cos(theta)
    new_node.y += distance * math.sin(theta)
    new_node.path_x.append(new_node.x)
    new_node.path_y.append(new_node.y)
    new_node.parent = from_node
    new_node.cost = from_node.cost + distance

    return new_node


def _set_parent(node, parent):
    node.parent = parent
    node.path_x = [parent.x, node.x]
    node.path_y = [parent.y, node.y]


def _propagate_cost_to_leaves(parent_node, all_list):
    for node in all_list:
        if node.parent is parent_node:
            node.cost = parent_node.cost + math.hypot(node.x - parent_node.x, node.y - parent_node.y)
            _propagate_cost_to_leaves(node, all_list)


def _is_ancestor(candidate, node):
    parent = node.parent
    while parent is not None:
        if parent is candidate:
            return True
        parent = parent.parent
    return False
