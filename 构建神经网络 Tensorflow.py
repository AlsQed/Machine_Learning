import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import BinaryCrossentropy

x_train, y_train = 0, 1

# 建立模型
# 顺序决定层中的单位数及函数
model = Sequential(
    [
        Dense(units=25, activation='sigmoid'),
        Dense(units=10, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')
    ]
)

# 编译模型
# 决定损失函数
model.compile(loss=BinaryCrossentropy())

# 训练模型
# 决定梯度下降步数--epochs
model.fit(x_train, y_train, epochs=5)

# 评估模型
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print('\nTest accuracy:', test_acc)
