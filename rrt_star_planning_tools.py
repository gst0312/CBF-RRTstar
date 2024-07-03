import copy

import cvxopt as cvx
import random
from log_reg import *
from matplotlib import pyplot as plt


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0


def cbf_rrt_star_steer(nearest_node, rnd_node, beta_opts, steps, v):
    t = 2 / steps  # 时间步长
    d, theta = calc_distance_and_angle(nearest_node, rnd_node)
    new_node_list = []
    from_node = nearest_node
    new_node = get_new_node(from_node, theta, d, t, v)

    for i in range(int(steps)):
        # CBF-QP Implementation
        x1 = new_node.x
        x2 = new_node.y

        # 设置CBF-QP算法的参数
        k1 = 4
        k2 = 2
        w_ref = 0.0
        g_mat = []  # 约束矩阵
        h_vec = []  # 约束向量

        for each in beta_opts:
            b_x = barrier_function(each, x1, x2)  # 计算障碍函数值
            B_dot = barrier_function_derivative(each, x1, x2, theta, v)  # 计算一阶导数
            B_ddot_c, B_ddot_w = barrier_function_second_derivative(each, x1, x2, theta, v)  # 计算二阶导数
            h_vec_value = B_ddot_c + k2 * B_dot + k1 * b_x
            g_mat.append([-B_ddot_w[0]])
            h_vec.append(h_vec_value[0])

        g_mat = np.append(g_mat, [[1], [-1]])
        h_vec = np.append(h_vec, [1.05, 1.05])
        P = cvx.matrix([[2.0]])
        q = cvx.matrix([w_ref])
        G = cvx.matrix(g_mat)
        h = cvx.matrix(h_vec)
        cvx.solvers.options['show_progress'] = False
        result = cvx.solvers.qp(P, q, G, h)

        w_sum = result['x'][0]
        theta += t * w_sum

        new_node = get_new_node(from_node, theta, d, t, v)
        d, _ = calc_distance_and_angle(from_node, new_node)
        new_node_list.append(new_node)
        from_node = new_node
        if d < t * v:
            break

    return new_node, new_node_list


def find_near_nodes(new_node, all_list, connect_circle_dist, expand_dis):
    nnode = len(all_list) + 1
    r = connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
    r = min(r, expand_dis)
    dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                 for node in all_list]
    near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
    return near_inds


def choose_parent(new_node, near_inds, all_list, beta_opts):
    if len(near_inds) == 0:
        return new_node

    # search nearest cost in near_inds
    costs = []
    for i in near_inds:
        new_node.parent = all_list[i]
        if check_collision(new_node, beta_opts):
            d, _ = calc_distance_and_angle(all_list[i], new_node)
            costs.append(d + all_list[i].cost)
        else:
            costs.append(float("inf"))  # the cost of collision node
    min_cost = min(costs)

    if min_cost == float("inf"):
        return new_node

    min_ind = near_inds[costs.index(min_cost)]
    new_node.parent = all_list[min_ind]
    new_node.cost = min_cost

    return new_node


def rewire(new_node, near_inds, all_list, beta_opts):
    for i in near_inds:
        near_node = all_list[i]
        d = math.sqrt((near_node.x - new_node.x) ** 2 + (near_node.y - new_node.y) ** 2)

        s_cost = new_node.cost + d

        if near_node.cost > s_cost:
            near_node_checking = copy.deepcopy(near_node)
            near_node_checking.parent = new_node
            if check_collision(near_node_checking, beta_opts):
                near_node.parent = new_node
                near_node.cost = s_cost


def search_best_goal_node(end, all_list, expand_dis, beta_opts):
    dist_to_goal_list = [
        calc_dist_to_goal(n, end) for n in all_list
    ]
    goal_inds = [
        dist_to_goal_list.index(i) for i in dist_to_goal_list
        if i <= expand_dis
    ]

    safe_goal_inds = []
    for goal_ind in goal_inds:
        end.parent = all_list[goal_ind]
        if check_collision(end, beta_opts):
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
    else:  # goal point sampling
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


def get_new_node(from_node, theta, d, t, v):
    new_node = Node(from_node.x, from_node.y)
    new_node.path_x.append(new_node.x)
    new_node.path_y.append(new_node.y)
    one_step = t * v
    if d < one_step:
        one_step = d
    new_node.x += one_step * math.cos(theta)
    new_node.y += one_step * math.sin(theta)
    d, _ = calc_distance_and_angle(from_node, new_node)
    new_node.cost = from_node.cost + d
    new_node.path_x.append(new_node.x)
    new_node.path_y.append(new_node.y)
    new_node.parent = from_node

    return new_node


# Barrier Functions and its first order and second order derivative #
def barrier_function(beta, x1, x2, power=4):
    # 检查输入是否是标量（单个浮点数）
    is_scalar = np.isscalar(x1) and np.isscalar(x2)
    # 如果是标量，将其转换为数组
    if is_scalar:
        x1, x2 = np.array([x1]), np.array([x2])
    # 检查输入是否是网格
    is_grid = x1.ndim == 2 and x2.ndim == 2
    # 如果是网格，暂时将其展平
    if is_grid:
        original_shape = x1.shape
        x1, x2 = x1.flatten(), x2.flatten()
    # 创建特征矩阵
    X = [np.ones_like(x1)]  # 添加偏置项
    for i in range(power + 1):
        for j in range(i + 1):
            X.append(np.power(x1, i - j) * np.power(x2, j))
    del X[1]  # 把100*100 的1阵删了
    # 将特征列表转换为矩阵
    X = np.array(X).T
    # 确保 safty_para 是正确的形状（列向量）
    beta_array = np.array(beta)
    if beta_array.ndim == 1:
        beta_array = beta_array.reshape(-1, 1)
    # 计算结果
    barrier_functions = np.dot(X, beta_array).flatten()
    # 如果输入是网格，将结果重塑为原始形状
    if is_grid:
        barrier_functions = barrier_functions.reshape(original_shape)
    # 如果输入是标量，返回单个值而不是数组
    if is_scalar:
        barrier_functions = barrier_functions[0]
    return barrier_functions


def barrier_function_derivative(beta, x1, x2, theta, v, power=4):
    # 计算速度在x和y方向上的分量
    v1 = v * math.cos(theta)
    v2 = v * math.sin(theta)
    # 将参数转换为数组
    beta_array = np.array([beta])
    # 初始化dx1，用于计算x1方向的导数
    dx1 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx1.append((i - j) * np.power(x1, i - j - 1) * np.power(x2, j))
    del dx1[0]  # 删除第一个元素
    dx1 = np.dot(v1, dx1)  # 计算在x1方向的导数
    # 初始化dx2，用于计算x2方向的导数
    dx2 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx2.append(j * np.power(x1, i - j) * np.power(x2, j - 1))
    del dx2[0]  # 删除第一个元素
    dx2 = np.dot(v2, dx2)  # 计算在x2方向的导数
    # 计算总导数
    dh = dx1 + dx2
    B_dot = np.dot(beta_array, dh)
    return B_dot


def barrier_function_second_derivative(beta, x1, x2, theta, v, power=4):
    # 将安全参数转换为数组
    beta_array = np.array([beta])
    # 计算速度在x和y方向上的分量
    v1 = v * math.cos(theta)
    v2 = v * math.sin(theta)
    # 初始化速度导数
    dv = 0
    dv1 = dv * math.cos(theta) - v * math.sin(theta)
    dv2 = dv * math.sin(theta) + v * math.cos(theta)
    # 计算x1方向的二阶导数
    dx1_dx1 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx1_dx1.append((i - j) * (i - j - 1) * np.power(x1, i - j - 2) * np.power(x2, j))
    del dx1_dx1[0]  # 删除第一个元素
    dx1_dx1 = np.dot(v1, np.dot(v1, dx1_dx1))
    # 计算x1和x2之间的二阶导数
    dx1_dx2 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx1_dx2.append((i - j) * j * np.power(x1, i - j - 1) * np.power(x2, j - 1))
    del dx1_dx2[0]  # 删除第一个元素
    dx1_dx2 = np.dot(v2, np.dot(v1, dx1_dx2))
    # 计算x1和theta之间的二阶导数
    dx1_dtheta = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx1_dtheta.append((i - j) * np.power(x1, i - j - 1) * np.power(x2, j))
    del dx1_dtheta[0]  # 删除第一个元素
    dx1_dtheta = np.dot(dv1, dx1_dtheta)
    # 计算x2和x1之间的二阶导数
    dx2_dx1 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx2_dx1.append(j * (i - j) * np.power(x1, i - j - 1) * np.power(x2, j - 1))
    del dx2_dx1[0]  # 删除第一个元素
    dx2_dx1 = np.dot(v2, np.dot(v1, dx2_dx1))
    # 计算x2方向的二阶导数
    dx2_dx2 = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx2_dx2.append(j * (j - 1) * np.power(x1, i - j) * np.power(x2, j - 2))
    del dx2_dx2[0]  # 删除第一个元素
    dx2_dx2 = np.dot(v2, np.dot(v2, dx2_dx2))
    # 计算x2和theta之间的二阶导数
    dx2_dtheta = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            dx2_dtheta.append(j * np.power(x1, i - j) * np.power(x2, j - 1))
    del dx2_dtheta[0]  # 删除第一个元素
    dx2_dtheta = np.dot(dv2, dx2_dtheta)
    # 计算所有二阶导数的和
    ddh_c = dx1_dx1 + dx1_dx2 + dx2_dx1 + dx2_dx2
    ddh_w = dx1_dtheta + dx2_dtheta
    # 计算结果
    ddB_c = np.dot(beta_array, ddh_c)
    ddB_w = np.dot(beta_array, ddh_w)
    return ddB_c, ddB_w


def calc_dist_to_goal(node, end):
    dx = node.x - end.x
    dy = node.y - end.y
    return math.hypot(dx, dy)


def check_collision(node, beta_opts):
    # 获取父节点和当前节点的坐标
    x1, y1 = node.parent.x, node.parent.y
    x2, y2 = node.x, node.y

    # 计算节点之间的直线距离
    lin = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 确定采样点的数量
    p = int(lin / 0.5)

    if x2 > x1:
        # 如果x2大于x1，生成x方向的采样点
        xlist = np.linspace(x1, x2, p)
        # 计算对应的y方向采样点
        ylist = ((y2 - y1) * (xlist - x2) / (x2 - x1)) + y2
    elif x2 < x1:
        # 如果x2小于x1，生成x方向的采样点
        xlist = np.linspace(x2, x1, p)
        # 计算对应的y方向采样点
        ylist = ((y2 - y1) * (xlist - x2) / (x2 - x1)) + y2
    else:
        # 如果x1和x2相等，只生成y方向的采样点
        if y2 > y1:
            ylist = np.linspace(y1, y2, p)
        else:
            ylist = np.linspace(y2, y1, p)
        xlist = x2 * np.ones_like(ylist)  # x方向采样点为常数

    # 遍历每个障碍参数
    for i in range(len(beta_opts)):
        # 计算障碍函数值
        ver_list = barrier_function(beta_opts[i], xlist, ylist)
        if np.isscalar(ver_list):
            # 如果结果是标量，直接比较是否大于0
            if not ver_list > 0:
                return False  # 如果不满足条件，返回False
        else:
            # 如果结果是数组，检查所有元素是否大于0
            if not np.all(ver_list > 0):
                return False  # 如果不满足条件，返回False

    return True  # 如果所有障碍函数值均满足条件，返回True


def draw_graph(start, end, min_xrand, max_xrand, min_yrand, max_yrand, all_list, obstacle_list, beta_opts, rnd=None):
    plt.clf()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    if rnd is not None:
        plt.plot(rnd.x, rnd.y, "^k")
    for node in all_list:
        if node.parent:
            plt.plot(node.path_x, node.path_y, "-g")

    for i in range(len(obstacle_list)):
        each = obstacle_list[i]
        each.append(each[0])  # repeat the first point to create a 'closed loop'
        xs, ys = zip(*each)  # create lists of x and y values
        plt.plot(xs, ys)
        draw_boundary(beta_opts[i])

    plt.plot(start.x, start.y, "xr")
    plt.plot(end.x, end.y, "xr")
    plt.axis("equal")
    plt.axis([min_xrand, max_xrand, min_yrand, max_yrand])
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.pause(0.01)


def generate_final_course(end, all_list, goal_ind):
    path = [[end.x, end.y]]
    node = all_list[goal_ind]
    while node.parent is not None:
        path.append([node.x, node.y])
        node = node.parent
    path.append([node.x, node.y])

    return path
