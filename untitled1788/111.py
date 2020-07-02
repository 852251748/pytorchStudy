import numpy as np

# rng = np.random.RandomState(0)
# x = 6 * rng.rand(100, 1)
# y = np.sin(x)
# y1 = np.sin(x).ravel()
#
# print(y.shape, y1.shape)
#
# print(rng.rand(x.shape[0] // 5).shape)
#
# print((2 * (rng.rand(x.shape[0] // 5) - 0.5)).shape)

a = np.array([[1, 2, 3], [3, 5, 9]])
print(a.shape)
print(a.mean(1))