import pandas as pd
import torch
from scipy.optimize import fmin_bfgs
from torch import nn
from torch.optim import LBFGS

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
def sig(theta, x):
    theta = np.ravel(theta)
    theta = np.array([theta])
    z = np.clip(np.dot(x, theta.T), -500, 500)  # 限制值的范围
    return 1 / (1 + np.exp(-z))


def Loss(theta, x, y):
    y_hat = sig(theta, x)
    m = np.shape(y)[0]
    epsilon = 1e-15  # 小常数
    return (-1 / m) * (np.dot(y.T, np.log(y_hat + epsilon)) + np.dot((1 - y.T), np.log(1 - y_hat + epsilon)))


def reg_Loss(theta, x, y, lamb=1):
    theta_1 = theta[1:]
    regularized_term = (lamb / (2 * len(x))) * np.power(theta_1, 2).sum()
    return Loss(theta, x, y) + regularized_term


def dLoss(theta, x, y):
    d = sig(theta, x) - y
    dt = np.dot(x.T, d)
    m = np.shape(y)[0]
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        grad[i] = (1 / m) * (dt[i][0])
    return grad


def dreg_Loss(theta, x, y, lamb=1):
    theta_1 = theta[1:]
    regularized_theta = (lamb / len(x)) * theta_1
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return dLoss(theta, x, y) + regularized_term


def h(theda, x1, x2, power=4):
    X = [1]
    for i in range(power + 1):
        for j in range(i + 1):
            X.append(np.power(x1, i - j) * np.power(x2, j))
    del X[1]
    return np.dot(X, theda.T)


def draw_boundary(theda):
    x1 = np.arange(0, 100, 1)
    x2 = np.arange(0, 100, 1)
    X1, X2 = np.meshgrid(x1, x2)
    z = h(theda, X1, X2)
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
