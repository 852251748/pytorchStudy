import torch
from torch import nn

if __name__ == '__main__':
    # # 二维数据就是图像
    # conv = nn.Conv2d(3, 16, 3, 1, padding=1)
    # x = torch.randn(1, 3, 10, 10)  # 格式是NCHW
    # # print(x)
    # print(conv(x).shape, conv(x))
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)

    target = torch.empty(3, dtype=torch.long).random_(5)
    print(input.shape,target.shape)
    output = loss(input, target)
    print(output)

# output.backward()