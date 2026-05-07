import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rrt_planning_tools import (  # noqa: E402
    Node,
    barrier_function_derivative,
    barrier_function_second_derivative,
    cbf_rrt_steer,
    check_collision,
    draw_graph,
)
from cbf_rrt import RRT  # noqa: E402
from rrt_star_planning_tools import Node as StarNode  # noqa: E402
from rrt_star_planning_tools import choose_parent, rewire  # noqa: E402


def test_cbf_steer_does_not_overshoot_near_sample():
    start = Node(0.0, 0.0)
    sample = Node(3.0, 0.0)

    new_node, new_nodes = cbf_rrt_steer(start, sample, [], steps=4, v=4.0)

    assert len(new_nodes) == 2
    assert new_node.x == 3.0
    assert abs(new_node.y) < 1e-12


def test_short_edge_collision_check_samples_endpoints():
    unsafe_half_space = [-0.1, 1.0] + [0.0] * 13
    start = Node(0.0, 0.0)
    end = Node(0.05, 0.0)
    end.parent = start

    assert check_collision(end, [unsafe_half_space]) is False


def test_barrier_derivatives_are_finite_at_origin():
    beta = [1.0] * 15

    first = barrier_function_derivative(beta, 0.0, 0.0, theta=0.0, v=1.0)
    second = barrier_function_second_derivative(beta, 0.0, 0.0, theta=0.0, v=1.0)

    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second[0]))
    assert np.all(np.isfinite(second[1]))


def test_draw_graph_does_not_mutate_obstacle_vertices():
    start = Node(0.0, 0.0)
    end = Node(1.0, 1.0)
    obstacles = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]

    draw_graph(start, end, 0, 2, 0, 2, [], obstacles, [])
    draw_graph(start, end, 0, 2, 0, 2, [], obstacles, [])

    assert len(obstacles[0]) == 3


def test_choose_parent_updates_edge_path_and_cost():
    root = StarNode(0.0, 0.0)
    far = StarNode(10.0, 0.0)
    far.parent = root
    far.cost = 10.0
    new_node = StarNode(1.0, 0.0)
    new_node.parent = far
    new_node.cost = 19.0

    chosen = choose_parent(new_node, [0, 1], [root, far], [])

    assert chosen.parent is root
    assert chosen.cost == 1.0
    assert chosen.path_x == [0.0, 1.0]
    assert chosen.path_y == [0.0, 0.0]


def test_rewire_updates_edge_path_and_descendant_costs():
    root = StarNode(10.0, 0.0)
    new_node = StarNode(0.0, 0.0)
    new_node.cost = 0.0

    old = StarNode(3.0, 0.0)
    old.parent = root
    old.cost = 10.0
    old.path_x = [root.x, old.x]
    old.path_y = [root.y, old.y]

    child = StarNode(4.0, 0.0)
    child.parent = old
    child.cost = 11.0
    child.path_x = [old.x, child.x]
    child.path_y = [old.y, child.y]

    rewire(new_node, [1], [root, old, child], [])

    assert old.parent is new_node
    assert old.cost == 3.0
    assert old.path_x == [0.0, 3.0]
    assert math.isclose(child.cost, 4.0)


def test_rrt_can_continue_after_first_goal_connection():
    plan = RRT(
        start=[0.0, 0.0],
        goal=[3.0, 0.0],
        obstacle_list=[],
        xrandArea=[0.0, 5.0],
        yrandArea=[0.0, 5.0],
        beta_opts=[],
        goal_sample_rate=100,
        max_iter=2,
        search_until_max_iter=True,
    )

    path = plan.cbf_rrt_planning(animation=False)

    assert path is not None
    assert path[0] == [3.0, 0.0]
