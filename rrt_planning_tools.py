import math
import random

import numpy as np
from log_reg import draw_boundary
from matplotlib import pyplot as plt


_EPS = 1e-9
_COLLISION_CHECK_RESOLUTION = 0.5
_TOTAL_STEER_TIME = 2.0


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None


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


def calc_distance_and_angle(from_node, to_node):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    return d, theta


def get_new_node(from_node, theta, distance):
    if distance < -_EPS:
        raise ValueError("distance must be non-negative")

    new_node = Node(from_node.x, from_node.y)
    new_node.path_x = [new_node.x]
    new_node.path_y = [new_node.y]
    new_node.x += max(0.0, distance) * math.cos(theta)
    new_node.y += max(0.0, distance) * math.sin(theta)
    new_node.path_x.append(new_node.x)
    new_node.path_y.append(new_node.y)
    new_node.parent = from_node

    return new_node


def cbf_rrt_steer(nearest_node, rnd_node, beta_opts, steps, v):
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

        # The QP is evaluated at the predicted next position before applying
        # the angular-velocity correction, matching the multi-step CBF steer.
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


def _solve_cbf_qp(beta_opts, x1, x2, theta, v, k1=4.0, k2=2.0, w_ref=0.0, w_bounds=(-1.05, 1.05)):
    constraints = []
    for beta in beta_opts:
        b_x = float(barrier_function(beta, x1, x2))
        b_dot = float(barrier_function_derivative(beta, x1, x2, theta, v)[0])
        b_ddot_c, b_ddot_w = barrier_function_second_derivative(beta, x1, x2, theta, v)
        rhs = float(b_ddot_c[0] + k2 * b_dot + k1 * b_x)
        constraints.append((-float(b_ddot_w[0]), rhs))

    return _solve_scalar_qp(constraints, w_ref, w_bounds)


def _solve_scalar_qp(constraints, w_ref, w_bounds):
    lower, upper = w_bounds
    for coeff, bound in constraints:
        if not np.isfinite(coeff) or not np.isfinite(bound):
            raise ValueError("CBF-QP produced a non-finite constraint")
        if abs(coeff) <= _EPS:
            if bound < -_EPS:
                raise ValueError("CBF-QP is infeasible")
            continue

        candidate = bound / coeff
        if coeff > 0:
            upper = min(upper, candidate)
        else:
            lower = max(lower, candidate)

    if lower > upper + _EPS:
        raise ValueError("CBF-QP is infeasible")

    return min(max(w_ref, lower), upper)


# Barrier Functions and their first and second derivatives.
def barrier_function(beta, x1, x2, power=4):
    is_scalar = np.isscalar(x1) and np.isscalar(x2)
    x1_array = np.asarray([x1], dtype=float) if is_scalar else np.asarray(x1, dtype=float)
    x2_array = np.asarray([x2], dtype=float) if is_scalar else np.asarray(x2, dtype=float)

    original_shape = None
    if x1_array.ndim == 2 and x2_array.ndim == 2:
        original_shape = x1_array.shape
        x1_array = x1_array.flatten()
        x2_array = x2_array.flatten()

    features = [
        np.power(x1_array, x_power) * np.power(x2_array, y_power)
        for x_power, y_power in _monomial_powers(power)
    ]
    feature_matrix = np.asarray(features).T
    beta_array = np.asarray(beta, dtype=float).reshape(-1, 1)

    if feature_matrix.shape[1] != beta_array.shape[0]:
        raise ValueError(
            f"beta has length {beta_array.shape[0]}, expected {feature_matrix.shape[1]} for power={power}"
        )

    barrier_values = np.dot(feature_matrix, beta_array).flatten()
    if original_shape is not None:
        barrier_values = barrier_values.reshape(original_shape)
    if is_scalar:
        return float(barrier_values[0])
    return barrier_values


def barrier_function_derivative(beta, x1, x2, theta, v, power=4):
    h_x, h_y, _, _, _ = _barrier_partials(beta, x1, x2, power)
    v1 = v * math.cos(theta)
    v2 = v * math.sin(theta)
    return np.array([h_x * v1 + h_y * v2], dtype=float)


def barrier_function_second_derivative(beta, x1, x2, theta, v, power=4):
    h_x, h_y, h_xx, h_xy, h_yy = _barrier_partials(beta, x1, x2, power)
    v1 = v * math.cos(theta)
    v2 = v * math.sin(theta)

    b_ddot_c = h_xx * v1 * v1 + 2.0 * h_xy * v1 * v2 + h_yy * v2 * v2
    b_ddot_w = h_x * (-v * math.sin(theta)) + h_y * (v * math.cos(theta))
    return np.array([b_ddot_c], dtype=float), np.array([b_ddot_w], dtype=float)


def _barrier_partials(beta, x1, x2, power):
    beta_array = np.asarray(beta, dtype=float).ravel()
    powers = _monomial_powers(power)
    if len(beta_array) != len(powers):
        raise ValueError(f"beta has length {len(beta_array)}, expected {len(powers)} for power={power}")

    h_x = 0.0
    h_y = 0.0
    h_xx = 0.0
    h_xy = 0.0
    h_yy = 0.0
    for coeff, (x_power, y_power) in zip(beta_array, powers):
        h_x += coeff * _partial_monomial(x1, x2, x_power, y_power, 1, 0)
        h_y += coeff * _partial_monomial(x1, x2, x_power, y_power, 0, 1)
        h_xx += coeff * _partial_monomial(x1, x2, x_power, y_power, 2, 0)
        h_xy += coeff * _partial_monomial(x1, x2, x_power, y_power, 1, 1)
        h_yy += coeff * _partial_monomial(x1, x2, x_power, y_power, 0, 2)

    return h_x, h_y, h_xx, h_xy, h_yy


def _monomial_powers(power):
    return [(total_power - y_power, y_power)
            for total_power in range(power + 1)
            for y_power in range(total_power + 1)]


def _partial_monomial(x1, x2, x_power, y_power, x_order, y_order):
    if x_power < x_order or y_power < y_order:
        return 0.0

    coeff = _falling_factorial(x_power, x_order) * _falling_factorial(y_power, y_order)
    return coeff * (x1 ** (x_power - x_order)) * (x2 ** (y_power - y_order))


def _falling_factorial(n, order):
    result = 1
    for value in range(n - order + 1, n + 1):
        result *= value
    return result


def calc_dist_to_goal(node, end):
    dx = node.x - end.x
    dy = node.y - end.y
    return math.hypot(dx, dy)


def get_final_node(node, end):
    d, theta = calc_distance_and_angle(node, end)
    return get_new_node(node, theta, d)


def check_collision(node, beta_opts, resolution=_COLLISION_CHECK_RESOLUTION):
    if node.parent is None:
        raise ValueError("collision checking requires node.parent")

    x1, y1 = node.parent.x, node.parent.y
    x2, y2 = node.x, node.y
    edge_length = math.hypot(x2 - x1, y2 - y1)

    if edge_length <= _EPS:
        xlist = np.array([x2], dtype=float)
        ylist = np.array([y2], dtype=float)
    else:
        sample_count = max(2, int(math.ceil(edge_length / resolution)) + 1)
        ratios = np.linspace(0.0, 1.0, sample_count)
        xlist = x1 + (x2 - x1) * ratios
        ylist = y1 + (y2 - y1) * ratios

    for beta in beta_opts:
        values = np.asarray(barrier_function(beta, xlist, ylist))
        if not np.all(values > 0):
            return False

    return True


def draw_graph(start, end, min_xrand, max_xrand, min_yrand, max_yrand, all_list, obstacle_list, beta_opts, rnd=None):
    plt.clf()
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    if rnd is not None:
        plt.plot(rnd.x, rnd.y, "^k")
    for node in all_list:
        if node.parent:
            plt.plot(node.path_x, node.path_y, "-g")

    for i, each in enumerate(obstacle_list):
        if len(each) == 0:
            continue
        closed_points = list(each)
        if each[0] != each[-1]:
            closed_points.append(each[0])
        xs, ys = zip(*closed_points)
        plt.plot(xs, ys)
        if i < len(beta_opts):
            draw_boundary(beta_opts[i])

    plt.plot(start.x, start.y, "xr")
    plt.plot(end.x, end.y, "xr")
    plt.axis("equal")
    plt.axis([min_xrand, max_xrand, min_yrand, max_yrand])
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    if "agg" not in plt.get_backend().lower():
        plt.pause(0.01)


def generate_final_course(end, all_list, goal_ind):
    path = [[end.x, end.y]]
    node = all_list[goal_ind]
    visited = set()
    while node.parent is not None:
        if id(node) in visited:
            raise RuntimeError("cycle detected in the generated tree")
        visited.add(id(node))
        path.append([node.x, node.y])
        node = node.parent
    path.append([node.x, node.y])

    return path
