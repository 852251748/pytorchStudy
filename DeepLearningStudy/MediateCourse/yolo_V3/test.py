import numpy as np
import torch


def oneHot(cls, i):
    label = np.zeros(cls)
    label[i] = 1
    return label


f = torch.nn.Conv2d(3, 32, 3, 2, 0)
input = torch.randn(1, 3, 416, 416)
y = f(input)
print(y.shape)
# print(a)
