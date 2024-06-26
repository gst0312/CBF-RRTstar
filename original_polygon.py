import numpy as np
import matplotlib.pyplot as plt
import math

# Expand the given polygon by a specified safe distance to create a buffer zone around it
def expand_poly(points, sd):
    points.append(points[0])  # add the first point to create a 'closed loop'
    x, y = zip(*points)  # create lists of x and y values

    # add the second point and make it an array
    x = np.append(x, x[1])
    y = np.append(y, y[1])

    # 初始化扩展后的顶点列表
    Qi_x_list = []
    Qi_y_list = []
    for i in range(len(x) - 2):
        # Calculate the edge length d1 between the current vertex and the next vertex, and the edge length
        # d2 between the next vertex of the current vertex and the vertex after that
        d1 = ((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) ** 0.5
        d2 = ((x[i + 1] - x[i + 2]) ** 2 + (y[i + 1] - y[i + 2]) ** 2) ** 0.5
        # 两条边的夹角
        ab = (x[i + 1] - x[i]) * (x[i + 1] - x[i + 2]) + (y[i + 1] - y[i]) * (y[i + 1] - y[i + 2])
        cosA = ab / (d1 * d2)
        sinA = (1 - cosA ** 2) ** 0.5

        # 判断凹凸点（叉积）
        P1P2_x = x[i + 1] - x[i]
        P1P2_y = y[i + 1] - y[i]
        P2P3_x = x[i + 2] - x[i + 1]
        P2P3_y = y[i + 2] - y[i + 1]
        P = (P1P2_y * P2P3_x) - (P1P2_x * P2P3_y)
        if P < 0:  # 凹点
            sinA = -sinA
        dv1 = sd / sinA
        v1_x = (dv1 / d1) * (x[i + 1] - x[i])
        v1_y = (dv1 / d1) * (y[i + 1] - y[i])
        v2_x = (dv1 / d2) * (x[i + 1] - x[i + 2])
        v2_y = (dv1 / d2) * (y[i + 1] - y[i + 2])
        PiQi_x = v1_x + v2_x
        PiQi_y = v1_y + v2_y
        Qi_x = PiQi_x + x[i + 1]
        Qi_x_list.append(Qi_x)
        Qi_y = PiQi_y + y[i + 1]
        Qi_y_list.append(Qi_y)

    Qi_x_list.append(Qi_x_list[0])
    Qi_y_list.append(Qi_y_list[0])

    new_points = np.asarray(list(zip(Qi_x_list, Qi_y_list)))
    return Qi_x_list, Qi_y_list, new_points

# Sample points along the edges of a polygon and add them to the list of vertices
def sample_points(x, y):
    # Initialize empty lists to store the additional sample points
    xadd = []
    yadd = []

    # Convert the input x and y coordinates to numpy arrays
    xs = np.asarray(x)
    ys = np.asarray(y)

    # Loop through each edge of the polygon (from each point to the next)
    for i in range(len(xs) - 1):
        if xs[i + 1] == xs[i]:  # Avoid division by zero
            xtemp = np.repeat(xs[i], int(abs(ys[i + 1] - ys[i])))
            ytemp = np.linspace(ys[i], ys[i + 1], int(abs(ys[i + 1] - ys[i])))
        else:
            # Calculate the slope of the current edge
            slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
            # Calculate the length of the edge
            length = math.sqrt(((ys[i + 1] - ys[i]) ** 2) + ((xs[i + 1] - xs[i]) ** 2))

            if slope > 1e5 or slope < -1e5:  # Check if the slope is very large
                # Calculate the inverse slope
                inverse_slope = (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
                # Generate sample points along the y direction
                ytemp = np.linspace(ys[i], ys[i + 1], int(length))
                xtemp = inverse_slope * (ytemp - ys[i]) + xs[i]
            else:
                # Generate sample points along the x direction
                xtemp = np.linspace(xs[i], xs[i + 1], int(length))
                ytemp = slope * (xtemp - xs[i]) + ys[i]

        # Append the sample points to the lists
        xadd = np.append(xadd, xtemp)
        yadd = np.append(yadd, ytemp)

    # Append the additional sample points to the original coordinates
    xs = np.asanyarray(np.append(xs, xadd))
    ys = np.asanyarray(np.append(ys, yadd))

    # Return the updated x and y coordinates
    return xs, ys

# Determine if a point is inside a polygon using the ray-casting algorithm.
def is_in_poly(p, points):
    x = np.zeros(len(points))
    px, py = p
    is_in = False
    for i, corner in enumerate(points):
        next_i = i + 1 if i + 1 < len(points) else 0
        x1, y1 = corner
        x2, y2 = points[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) <= py <= max(y1, y2):  # find horizontal edges of polygon
            x[i] = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x[i] == px:  # if point is on edge
                is_in = True
                break
            elif x[i] > px:  # if point is on left-side of line
                if x[i] != x[i - 1]:
                    is_in = not is_in
    return is_in

# Generate a grid of points and determine if each point is inside the given polygon.
def draw_poly(obs, sd):
    new_all_points = []
    all_xs = []
    all_ys = []

    for points in obs:
        x, y, new_points = expand_poly(points, sd)
        xs, ys = sample_points(x, y)
        new_all_points.append(new_points)
        all_xs.extend(xs)
        all_ys.extend(ys)

    # Generate initial x and y coordinates for the grid
    x_init = np.arange(0, 105, 1)
    y_init = np.arange(0, 84, 1)

    # Create a meshgrid from the initial x and y coordinates
    x_p, y_p = np.meshgrid(x_init, y_init)
    x_p = np.ravel(x_p)
    y_p = np.ravel(y_p)

    # Initialize an array to store the classification values
    p_value = np.zeros_like(x_p)

    # Loop through each grid point to determine if it is inside any of the polygons
    for i in range(len(x_p)):
        p = [x_p[i], y_p[i]]
        for new_points in new_all_points:
            if is_in_poly(p, new_points):
                p_value[i] = 1
                break

    # Append the original polygon vertices to the grid points
    x_p = np.append(x_p, all_xs)
    y_p = np.append(y_p, all_ys)

    # Create an array of ones with the same length as the number of original vertices
    p_s = np.ones(len(all_xs))

    # Append the classification values for the original vertices to the grid points
    p_value = np.append(p_value, p_s)

    # 绘制图形
    plt.figure(figsize=(10, 8))
    for new_points in new_all_points:
        plt.plot(new_points[:, 0], new_points[:, 1], 'r-')
    plt.scatter(x_p[p_value == 0], y_p[p_value == 0], c='blue', label='Free Space', s=1)
    plt.scatter(x_p[p_value == 1], y_p[p_value == 1], c='red', label='Obstacle', s=1)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Return the updated grid points and their classification values
    return x_p, y_p, p_value
