# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost分类器
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 训练模型
xgb_classifier.fit(X_train, y_train)

# 进行预测
y_pred = xgb_classifier.predict(X_test)

# 打印混淆矩阵和分类报告
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_classifier.feature_importances_, y=iris.feature_names)
plt.title('Feature Importance from XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()