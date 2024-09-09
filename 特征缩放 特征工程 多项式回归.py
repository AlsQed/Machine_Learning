# 特征缩放
# 特征缩放是数据预处理的一个重要步骤,目的是将不同量纲的特征值映射到一个统一的范围内,通常是 [0, 1] 或 [-1, 1]。这样可以避免某些特征因量纲不同而对模型产生不均衡的影响。常用的方法有:
#
# 标准化(Z-score归一化)：将特征值减去均值,再除以标准差。得到均值为0,方差为1的特征。
# 最小-最大归一化(Min-Max归一化)：将特征值减去最小值,再除以最大值与最小值的差。得到 [0, 1] 范围内的特征。

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# 假设有一个包含身高和体重的数据集
df = pd.DataFrame({'height': [160, 165, 170, 175, 180],
                   'weight': [55, 60, 65, 70, 75]})

# 标准化
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)
print(df_standardized)

# 最小-最大归一化
scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df)
print(df_minmax)

# 特征工程
# 特征工程是指根据业务需求和数据特点,对原始特征进行组合、变换、筛选等操作,以期得到更加有效的特征。常见的特征工程技术包括:
#
# 衍生特征:根据原始特征创造新的特征,如年龄-出生年、身高/体重。
# 特征交叉:将两个或多个特征进行组合,如年龄收入、男性已婚。
# 特征离散化:将连续特征离散化为分类特征,如年龄离散化为青年/中年/老年。
# 特征选择:根据特征与目标变量的相关性,选择最有预测能力的特征子集。

# 多项式回归
# 多项式回归是一种非线性回归模型,可以拟合更复杂的函数关系。它通过在原有特征基础上增加高次项特征,来捕捉输入与输出之间的非线性关系。
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 生成一些非线性数据
X = np.arange(1, 11).reshape(-1, 1)
y = np.sin(X).ravel()

# 创建多项式特征
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# 训练多项式回归模型
model = LinearRegression()
model.fit(X_poly, y)

# 预测
y_pred = model.predict(X_poly)
print(y_pred)