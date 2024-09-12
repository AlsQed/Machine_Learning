import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 创建示例数据
data = {
    'Feature1': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Feature2': [1, 2, 1, 2, 1, 2, 1, 2],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 独热编码
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Feature1']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Feature1']))

# 合并特征
X = pd.concat([encoded_df, df[['Feature2']]], axis=1)
y = df['Target']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建和训练决策树分类模型（使用熵作为标准）
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估结果
print(classification_report(y_test, y_pred))
print("准确率:", accuracy_score(y_test, y_pred))

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['0', '1'], filled=True)
plt.title("决策树可视化")
plt.show()