# 梯度下降公式
# 假设代价函数为J(w,b)
# temp_w = w - a*[J(w,b)对w求偏导]
# temp_b = b - a*[J(w,b)对b求偏导]
# w = temp_w
# b = temp_b
# a为学习率(始终为正),temp_w与w不可交换顺序
# 该算法可找到代价函数对应图像最低点
# 当到达最低点时,代价函数求导为0,即此时无论变化率为多少,w不再改变
# 梯度下降指向区域最小值而非整体最小值
# 不同变化率可能指向不同最小值
# 线性回归模型只有一个最小值
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成随机数据
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 3 + np.random.randn(100, 1)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X, y)

# 获取模型参数
# 获取模型的截距参数
intercept = model.intercept_[0]
# 获取模型的斜率参数
slope = model.coef_[0][0]

# 可视化训练数据和模型拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Training Data')
plt.plot(X, intercept + slope * X, color='r', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

# 可视化梯度下降过程
# 设置梯度下降的迭代次数
iterations = 1000
# 设置梯度下降的学习率
learning_rate = 0.01
# 随机初始化模型参数
theta = np.random.rand(2, 1)

# 储存每次迭代的损失函数值
costs = []
for i in range(iterations):
    y_pred = theta[0] + theta[1] * X
    gradient = np.array([[np.mean(y_pred - y)], [np.mean((y_pred - y) * X.ravel())]])
    theta = theta - learning_rate * gradient
    cost = np.mean((y_pred - y) ** 2)
    costs.append(cost)

plt.figure(figsize=(8, 6))
plt.plot(range(iterations), costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.show()
