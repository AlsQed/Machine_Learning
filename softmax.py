from keras.src.datasets import mnist
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential


# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28*28)  # 将图像展平
x_test = x_test.reshape(-1, 28*28)

# 将标签进行独热编码
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# 构建神经网络模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Softmax 输出层
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=100)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'测试损失: {test_loss}, 测试准确率: {test_accuracy}')