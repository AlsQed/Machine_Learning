import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# 生成数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 多项式特征
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)

# Ridge回归
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_poly, y)

# 可视化
X_new = np.linspace(0, 2, 100).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
y_predict = ridge_reg.predict(X_new_poly)

plt.scatter(X, y, s=10)
plt.plot(X_new, y_predict, "r-", linewidth=2)
plt.title("Ridge Regression (Regularized Linear Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.title('Regularized Linear Regression')
plt.axis([0, 2, 0, 15])
plt.show()

from sklearn.linear_model import LogisticRegression

# 生成数据
np.random.seed(0)
X = np.random.rand(100, 2) * 2 - 1  # 范围在[-1, 1]
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 线性可分

# 逻辑回归
log_reg = LogisticRegression(C=1, solver='liblinear', penalty='l2')
log_reg.fit(X, y)

# 可视化
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Logistic Regression with Regularization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()