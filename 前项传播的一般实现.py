import numpy as np
import tensorflow as tf
from keras import layers,models


def dense(a_in, W, b, g):
    """
    计算全连接层的输出。

    参数:
    a_in -- 输入激活值,一个形状为 (n_in,) 的 NumPy 数组
    W -- 权重矩阵,一个形状为 (n_in, n_out) 的 NumPy 数组
    b -- 偏置向量,一个形状为 (n_out,) 的 NumPy 数组
    g -- 激活函数,一个 Python 函数

    返回:
    a_out -- 输出激活值,一个形状为 (n_out,) 的 NumPy 数组
    """
    # 获取输出单元(神经元)的数量
    units = W.shape[1]

    # 初始化输出激活值为 0
    a_out = np.zeros(units)

    # 对每个输出单元进行计算
    for j in range(units):
        # 获取第 j 列的权重向量
        w = W[:, j]

        # 计算线性激活值
        z = np.dot(w, a_in) + b[j]

        # 应用激活函数获得输出激活值
        a_out[j] = g(z)

    return a_out
