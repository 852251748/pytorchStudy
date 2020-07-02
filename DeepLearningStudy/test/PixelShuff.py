import numpy as np

# [[]]

a = np.array([
    [
        [1, 1],
        [1, 1]
    ],
    [
        [2, 2],
        [2, 2]
    ],
    [
        [3, 3],
        [3, 3]
    ],
    [
        [4, 4],
        [4, 4]
    ]
])
b = np.array([[1, 2, 1, 2],
              [3, 4, 3, 4],
              [1, 2, 1, 2],
              [3, 4, 3, 4]])
print(a.shape, b.shape)

# print(a)

x = a.reshape(4, 4)
x = x.transpose(1, 0)
x = x.reshape(2, 4, 2)
x = x.transpose(1, 0, 2)
x = x.reshape(4, 4)

print(x, x.shape)
