# 建立模型
import torch
from torch import nn


class NetV1(nn.Module):

    def __init__(self):
        super().__init__()
        # 使用正太分布创建W 权重一般符合正太分布
        self.W = nn.Parameter(torch.randn(784, 10))

    def forward(self, x):
        # 输入乘以权重
        h = x @ self.W
        # 做下softmax处理
        h = torch.exp(h)

        # 求下分母
        z = torch.sum(h, dim=1, keepdim=True)

        return h / z


class NetV2(nn.Module):

    def __init__(self):
        super().__init__()
        # 使用正太分布创建W 权重一般符合正太分布
        self.fc1 = nn.Linear(784, 100)
        self.reul = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入乘以权重
        h = self.fc1(x)
        h = self.reul(h)
        h = self.fc2(h)
        y = self.softmax(h)

        return y


class NetV3(nn.Module):

    def __init__(self):
        super().__init__()
        # 使用正太分布创建W 权重一般符合正太分布
        self.sequential = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.ReLU(),
            nn.Linear(2, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 输入乘以权重
        y = self.sequential(x)

        return y


if __name__ == "__main__":
    net = NetV1()

    x = torch.rand(4, 784)

    print(net.forward(x).shape, net.forward(x))
