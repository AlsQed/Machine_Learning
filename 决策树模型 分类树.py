import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 创建示例数据
data = {
    'Feature1': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Feature2': [1, 2, 1, 2, 1, 2],
    'Target': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 独热编码
# 将每个类别转换为一个二进制特征，以便决策树可以有效地处理这些特征
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Feature1']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Feature1']))

# 合并特征
X = pd.concat([encoded_df, df[['Feature2']]], axis=1)
y = df['Target']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建和训练决策树分类模型
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估结果
print(classification_report(y_test, y_pred))
print("准确率:", accuracy_score(y_test, y_pred))