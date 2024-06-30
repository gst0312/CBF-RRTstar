import pandas as pd
from scipy.optimize import fmin_bfgs

from original_polygon import *


def constr_dataframe(x_p, y_p, p_value, power=4):
    data = {"f{}{}".format(i - p, p): np.power(x_p, i - p) * np.power(y_p, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)
            }
    data = pd.DataFrame(data)
    data.insert(len(data.columns), 'c_value', p_value)
    return data


def creat_init_feat_matrix(data):
    num_col = data.shape[1]
    features = data.iloc[:, 0:num_col - 1]
    target = data.iloc[:, num_col - 1:num_col]
    features = np.array(features.values)
    target = np.array(target.values)

    beta = np.zeros(num_col - 1)
    return features, target, beta


######################## For logistic regression #######################
def sig(beta, x):
    beta = np.ravel(beta)
    beta = np.array([beta])
    z = np.clip(np.dot(x, beta.T), -500, 500)  # 限制值的范围
    return 1 / (1 + np.exp(-z))


def Loss(beta, x, y):
    y_hat = sig(beta, x)
    m = np.shape(y)[0]
    epsilon = 1e-15  # 小常数
    return (-1 / m) * (np.dot(y.T, np.log(y_hat + epsilon)) + np.dot((1 - y.T), np.log(1 - y_hat + epsilon)))


def reg_Loss(beta, x, y, lamb=1):
    theta_1 = beta[1:]
    regularized_term = (lamb / (2 * len(x))) * np.power(theta_1, 2).sum()
    return Loss(beta, x, y) + regularized_term


def dLoss(beta, x, y):
    d = sig(beta, x) - y
    dt = np.dot(x.T, d)
    m = np.shape(y)[0]
    grad = np.zeros(len(beta))
    for i in range(len(beta)):
        grad[i] = (1 / m) * (dt[i][0])
    return grad


def dreg_Loss(beta, x, y, lamb=1):
    beta_1 = beta[1:]
    regularized_theta = (lamb / len(x)) * beta_1
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return dLoss(beta, x, y) + regularized_term


def h(beta, x1, x2, power=4):
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
    del X[1]  # 删除第二个元素，保持与原函数一致

    # 将特征列表转换为矩阵
    X = np.array(X).T

    # 确保 beta 是正确的形状（列向量）
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)

    # 计算结果
    result = np.dot(X, beta).flatten()

    # 如果输入是网格，将结果重塑为原始形状
    if is_grid:
        result = result.reshape(original_shape)

    return result


def draw_boundary(beta):
    x1 = np.arange(0, 100, 1)
    x2 = np.arange(0, 100, 1)
    X1, X2 = np.meshgrid(x1, x2)
    z = h(beta, X1, X2)
    plt.contour(X1, X2, z, [0])


def multi_classify(obs_list, sd):
    beta_opts = []
    for points in obs_list:
        x_p, y_p, p_value, _ = each_poly(points, sd)

        data = constr_dataframe(x_p, y_p, p_value)
        features, label, beta = creat_init_feat_matrix(data)
        beta_opt_i = fmin_bfgs(reg_Loss, beta, dreg_Loss, args=(features, label))
        beta_opt_i = -1 * np.ravel(beta_opt_i)
        beta_opts.append(beta_opt_i)

    draw_poly(obs_list, sd)
    for i in range(len(beta_opts)):
        draw_boundary(beta_opts[i])
    plt.xlim([-10, 110])
    plt.ylim([-10, 90])
    plt.show()

    return beta_opts
