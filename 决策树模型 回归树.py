import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 创建示例数据
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Target': [1.5, 2.5, 3.5, 4.5, 5.5]
}

df = pd.DataFrame(data)

# 拆分数据集
X = df[['Feature1', 'Feature2']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建和训练决策树回归模型
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# 进行预测
y_pred = regressor.predict(X_test)

# 评估结果
mse = mean_squared_error(y_test, y_pred)
print("均方误差:", mse)