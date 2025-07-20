import numpy as np

x = np.array([1, 2, 3])

print(x.__class__)
print(x.shape)
print(x.ndim)

W = np.array([[1, 2, 3],
              [4, 5, 6]])
print(W.ndim)
print(W.shape)

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

print(W + X)
print('---')
print(W * X)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b)
print(y)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Y = np.dot(A, B)
print(Y)