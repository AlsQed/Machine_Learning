# 拥有多个特征的线性回归模型——多元线性回归
# 当训练集有n个特征(即x),对应的模型参数也有n个
# f_w,b(x_n) = w_1*x_1 + w_2*x_2 +...+ w_n*x_n + b
# 其中w,x可被视作对应的向量
# 则模型可被缩写为f_w,b(x_n) = _w(向量w) * _x(向量x) + b 即向量点积形式

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义向量组
w = np.array([1, 2, 3])
b = 4
x = np.array([5, 6, 7])

# 使用numpy向量点积
f1 = np.dot(w, x) + b  # 向量化
# print(f1)

# 等价于
f2 = 0
for i in range(0, 3):  # 未向量化
    f2 = f2 + w[i] * x[i]
f2 = f2 + b
# print(f2)

# 生成模拟数据
np.random.seed(0)
m = 100
X1 = np.random.uniform(-10, 10, m)
X2 = np.random.uniform(-10, 10, m)
y = 2 * X1 + 3 * X2 + 10 + np.random.normal(0, 5, m)
# np.ones(m) 创建了一个长度为 m 的全 1 数组,这相当于在模型中添加了一个常数项
# 多元线性回归模型的表达式为:
# y = θ_0 + θ_1*x_1 + θ_2*x_2 + ... + θ_n*x_n
# np.column_stack() 函数将这三个数组按列的方式堆叠起来,形成一个 m x 3 的二维数组 X。这个矩阵 X 就是我们多元线性回归模型的输入特征矩阵。
# 每一行对应一个样本
# 每一列对应一个特征
X = np.column_stack((np.ones(m), X1, X2))


# 定义代价函数
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    return 1 / (2 * m) * np.sum((h - y) ** 2)


# 梯度下降算法
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = np.dot(X, theta)
        # X.T表示X的转置矩阵
        theta = theta - (alpha / m) * np.dot(X.T, h - y)
        J_history[i] = cost_function(X, y, theta)

    return theta, J_history


# 初始化模型参数
theta = np.zeros(3)

# 运行梯度下降算法
alpha = 0.01
num_iters = 1000
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

print("最终模型参数:")
print(theta)

# 绘制损失函数变化
plt.figure(figsize=(8, 6))
plt.plot(np.arange(num_iters), J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence of Gradient Descent')
plt.show()

# 绘制预测结果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, c='b', marker='o', label='Actual')

x1 = np.linspace(X1.min(), X1.max(), 50)
x2 = np.linspace(X2.min(), X2.max(), 50)
X1, X2 = np.meshgrid(x1, x2)
y_pred = theta[0] + theta[1] * X1 + theta[2] * X2
ax.plot_surface(X1, X2, y_pred, alpha=0.5, color='r', label='Prediction')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D Plot of the Data and Prediction')
ax.legend()
plt.show()
