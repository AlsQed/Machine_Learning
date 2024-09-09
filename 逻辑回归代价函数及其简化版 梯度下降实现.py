import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))


# 梯度下降
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    # Python中，@符号被称为“矩阵乘法运算符”。它用于执行两个矩阵之间的乘法操作
    for i in range(iterations):
        h = sigmoid(X @ theta)
        theta -= (alpha / m) * (X.T @ (h - y))
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# 数据生成
# make_classification 是 Scikit-learn 库中的一个函数，用于生成一个用于分类任务的随机数据集
# n_samples: 要生成的样本数量
# n_features: 特征的总数量（包括有用和无用的特征）
# n_informative: 信息量特征的数量（对分类有贡献）
# n_redundant: 冗余特征的数量，这些特征是由信息特征线性组合生成的
# n_clusters_per_class: 每个类别的簇数
# random_state: 随机数生成器的种子，用于结果复现
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
# hstack 是 NumPy 库中的一个函数，用于将多个数组沿水平方向（即列的方向）进行堆叠
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加偏置项
theta_initial = np.zeros(X.shape[1])

# 训练模型
alpha = 0.1
iterations = 1000
theta_final, cost_history = gradient_descent(X, y, theta_initial, alpha, iterations)

# 可视化
plt.scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolor='k', s=20)
x_value = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_value = -(theta_final[0] + theta_final[1] * x_value) / theta_final[2]
plt.plot(x_value, y_value, color='black')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 打印最终参数和准确率
predictions = sigmoid(X @ theta_final) >= 0.5
accuracy = accuracy_score(y, predictions)
print(f"Final Parameters: {theta_final}")
print(f"Accuracy: {accuracy * 100:.2f}%")