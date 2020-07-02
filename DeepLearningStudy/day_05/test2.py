import torch
import numpy as np

a = torch.tensor([[1., 2, 3, 4],
                  [2., 5, 6, 7],
                  [3., 3, 5, 8],
                  [4., 2, 1, 4]])

b = torch.tensor([[2., 3, 5, 2],
                  [3., 5, 6, 3],
                  [4., 5, 3, 9],
                  [3., 7, 5, 9]])

c = torch.tensor([[2.],
                  [3.],
                  [4.],
                  [5.]])
mask = c > 3
index = mask.nonzero()[:, 0]

c = a[index]
d = b[index]
print(c, d)
print((c - d))

print(torch.mean((c - d) ** 2))
