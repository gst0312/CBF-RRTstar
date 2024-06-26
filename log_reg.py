import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def feature_mapping(x, y, power=4, as_ndarray=False):
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)

def constr_dataframe(x_p, y_p, p_value, power=4):
    data = feature_mapping(x_p, y_p, power)
    data = pd.DataFrame(data)
    data.insert(len(data.columns), 'c_value', p_value)
    return data

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


def train_model(data, num_epochs=1000, learning_rate=0.01, lambda_reg=1):
    # 提取特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # 转换为张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 模型实例化
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()

        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def plot_decision_boundary(model, power):
    x1 = np.linspace(0, 100, 200)
    x2 = np.linspace(0, 100, 200)
    X1, X2 = np.meshgrid(x1, x2)
    mapped_features = feature_mapping(X1.ravel(), X2.ravel(), power, as_ndarray=True)
    X_mapped = torch.tensor(mapped_features, dtype=torch.float32)

    with torch.no_grad():
        Z = model(X_mapped).reshape(X1.shape)

    plt.contour(X1, X2, Z, levels=[0.5])
    plt.xlim([-10, 110])
    plt.ylim([-10, 90])
    plt.xlabel('x_1', fontsize=18)
    plt.ylabel('x_2', fontsize=18)
    plt.show()
