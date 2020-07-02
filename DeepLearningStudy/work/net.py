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
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        self.liner = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 输入乘以权重
        feature = self.sequential(x)
        y = self.liner(feature)
        return y, feature


class NetV4(nn.Module):

    def __init__(self):
        super().__init__()
        self.Sequential1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 16, 3, 2, 1, bias=False),
            nn.ReLU()
        )

        self.liner_layer = nn.Linear(16 * 4 * 4, 2, bias=False)
        self.output_layer = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        y = self.Sequential1(x)
        y = y.reshape(-1, 16 * 4 * 4)
        feature = self.liner_layer(y)
        output = self.output_layer(feature)
        return feature, output


if __name__ == "__main__":
    net = NetV4()

    x = torch.rand(2, 1, 28, 28)
    feature, output = net(x)

    print(feature.shape, output.shape)
