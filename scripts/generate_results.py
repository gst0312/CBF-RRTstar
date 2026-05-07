from pathlib import Path
import os
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "cbf_rrtstar_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(ROOT))

from cbf_rrt import RRT
from cbf_rrt_star import RRTStar
from log_reg import draw_boundary, multi_classify
from original_polygon import draw_poly
from rrt_planning_tools import draw_graph as draw_rrt_graph
from rrt_star_planning_tools import draw_graph as draw_rrt_star_graph


RESULTS_DIR = ROOT / "results"
START = [9, 8]
GOAL = [74.5, 68]
XRANGE = [0, 100]
YRANGE = [0, 80]
SAFE_DISTANCE = 4
RRT_MAX_ITER = 1200
RRT_STAR_FIRST_PATH_MAX_ITER = 500
RRT_STAR_MAX_ITER = 1000
RRT_STAR_EXPAND_DIS = 4.0
RRT_STAR_CONNECT_CIRCLE_DIST = 80.0

MULTI_OBSTACLES = [
    [[5, 71], [18, 74], [21, 64], [7, 62]],
    [[15, 40], [25, 47], [25, 38]],
    [[49, 46], [51, 56], [60, 54], [57, 45]],
    [[88, 32], [93, 35], [94, 30]],
    [[50, 17], [56, 20], [57, 16], [52, 14]],
]

PLANNING_OBSTACLES = [
    [[30, 20], [30, 50], [50, 60], [60, 20]],
]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(7)
    np.random.seed(7)

    multi_betas = multi_classify(_copy_obstacles(MULTI_OBSTACLES), SAFE_DISTANCE)
    _save_origin_obstacles(_copy_obstacles(MULTI_OBSTACLES))
    _save_barrier_plot(_copy_obstacles(MULTI_OBSTACLES), multi_betas)

    planning_betas = multi_classify(_copy_obstacles(PLANNING_OBSTACLES), SAFE_DISTANCE)

    random.seed(11)
    _save_rrt(planning_betas, "CBF_RRT.png")

    random.seed(13)
    _save_rrt_star(
        planning_betas,
        "star_not_max.png",
        search_until_max_iter=False,
        max_iter=RRT_STAR_FIRST_PATH_MAX_ITER,
    )

    random.seed(17)
    _save_rrt_star(
        planning_betas,
        "star_max.png",
        search_until_max_iter=True,
        max_iter=RRT_STAR_MAX_ITER,
    )

    random.seed(17)
    _save_rrt_star(
        planning_betas,
        "CBF_RRTstar.png",
        search_until_max_iter=True,
        max_iter=RRT_STAR_MAX_ITER,
    )


def _copy_obstacles(obstacles):
    return [[point[:] for point in obstacle] for obstacle in obstacles]


def _save_origin_obstacles(obstacles):
    plt.figure(figsize=(10, 8))
    draw_poly(obstacles, SAFE_DISTANCE)
    plt.xlim([-10, 110])
    plt.ylim([-10, 90])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "originobs.png", dpi=100)
    plt.close()


def _save_barrier_plot(obstacles, beta_opts):
    plt.figure(figsize=(10, 8))
    draw_poly(obstacles, SAFE_DISTANCE)
    for beta in beta_opts:
        draw_boundary(beta)
    plt.xlim([-10, 110])
    plt.ylim([-10, 90])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multi_classify.png", dpi=100)
    plt.close()


def _save_rrt(beta_opts, filename):
    plan = RRT(
        start=START,
        goal=GOAL,
        obstacle_list=_copy_obstacles(PLANNING_OBSTACLES),
        xrandArea=XRANGE,
        yrandArea=YRANGE,
        beta_opts=beta_opts,
        max_iter=RRT_MAX_ITER,
        search_until_max_iter=True,
    )
    path = plan.cbf_rrt_planning(animation=False)
    if path is None:
        raise RuntimeError("CBF-RRT failed to find a path")

    plt.figure(figsize=(10, 8))
    draw_rrt_graph(
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
    _plot_path(path)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=100)
    plt.close()


def _save_rrt_star(beta_opts, filename, search_until_max_iter, max_iter):
    plan = RRTStar(
        start=START,
        goal=GOAL,
        obstacle_list=_copy_obstacles(PLANNING_OBSTACLES),
        xrandArea=XRANGE,
        yrandArea=YRANGE,
        beta_opts=beta_opts,
        search_until_max_iter=search_until_max_iter,
        connect_circle_dist=RRT_STAR_CONNECT_CIRCLE_DIST,
        expand_dis=RRT_STAR_EXPAND_DIS,
        max_iter=max_iter,
    )
    path = plan.cbf_rrt_star_planning(animation=False)
    if path is None:
        raise RuntimeError(f"CBF-RRT* failed to find a path for {filename}")

    plt.figure(figsize=(10, 8))
    draw_rrt_star_graph(
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
    _plot_path(path)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=100)
    plt.close()


def _plot_path(path):
    plt.plot([x for (x, _) in path], [y for (_, y) in path], "-r", linewidth=2.0)
    plt.grid(True)


if __name__ == "__main__":
    main()
