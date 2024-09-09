import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.array([400, 500, 600, 700])
y = np.array([150000, 200000, 250000, 300000])

# 绘制数据点
plt.scatter(x, y)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (USD)')
plt.title('Housing Price vs Area')
# plt.show()

# 线性回归模型拟合
theta = np.polyfit(x, y, 1)
# np.polyfit() 函数是 NumPy 中用于拟合一元多项式的函数。
# np.polyfit(x, y, deg)
# x: 自变量数据,一维 NumPy 数组。
# y: 因变量数据,一维 NumPy 数组。
# deg: 多项式的阶数。当 deg=1 时,表示拟合一次线性多项式,即线性回归。
h = theta[0] * x + theta[1]

# 绘制拟合直线
plt.scatter(x, y)
plt.plot(x, h, color='r')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (USD)')
plt.title('Housing Price vs Area')
plt.show()

# 可视化代价函数
# 创建 100 个 theta0 值,范围从 -10000 到 10000。
theta0 = np.linspace(0, 1000, 1000)
theta1 = np.linspace(0, 1000, 1000)
# 初始化
J = np.zeros((len(theta0), len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        h = theta1[j] * x + theta0[i]
        J[i, j] = 1 / (2 * len(x)) * np.sum((h - y) ** 2)

plt.figure(figsize=(8, 6))
plt.contour(theta0, theta1, J, levels=50, cmap='viridis')
plt.clabel(plt.contour(theta0, theta1, J, levels=50), inline=True, fontsize=8)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title('Cost Function')
plt.show()
