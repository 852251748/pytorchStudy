import numpy as np

a = np.array([
    [
        [1, 1],
        [1, 1]
    ], [
        [2, 2],
        [2, 2]
    ], [
        [3, 3],
        [3, 3]
    ], [
        [4, 4],
        [4, 4]
    ]
])
x = a.reshape(2, 2, 2, 2)
print(x)
x = x.transpose(2, 0, 3, 1)
print(x)
x = x.reshape(4, 4)
print(x)
