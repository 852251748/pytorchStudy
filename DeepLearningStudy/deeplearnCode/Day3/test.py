import torch
from torch import nn
from torchvision import models


# SPP(空间金字塔池化)
# if __name__ == '__main__':
#     seq1 = nn.MaxPool2d(9, 8)
#     seq2 = nn.MaxPool2d(13, 13)
#     seq3 = nn.MaxPool2d(26, 26)
#
#     x = torch.randn(1, 256, 26, 26)
#     out1 = seq1(x)
#     out1 = out1.reshape(1, 256, -1)
#     out2 = seq2(x)
#     out2 = out2.reshape(1, 256, -1)
#     out3 = seq3(x)
#     out3 = out3.reshape(1, 256, -1)
#
#     print(out1.shape, out2.shape, out3.shape)
#
#     print(torch.cat((out1, out2, out3), dim=2).shape)
