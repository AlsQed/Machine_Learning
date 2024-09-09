import numpy as np
from keras.src.utils.module_utils import scipy

# 创建 2x3 矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# 创建 3x2 矩阵
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 矩阵乘法
C = np.dot(A, B)
# 或者使用 @ 运算符
C = A @ B

# 矩阵转置
A_T = A.T

# 矩阵求逆
A_inv = np.linalg.inv(A)

# 矩阵求特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 矩阵求行列式
det_A = np.linalg.det(A)

# 矩阵范数计算
# 计算 L2 范数
norm_A = np.linalg.norm(A)

# 计算 Frobenius 范数
frobenius_norm_A = np.linalg.norm(A, 'fro')

# 矩阵分解
# 奇异值分解
U, s, Vt = np.linalg.svd(A)

# LU 分解
P, L, U = scipy.linalg.lu(A)

# 矩阵求解线性方程组
# 求解 Ax = b
# x = np.linalg.solve(A, b)