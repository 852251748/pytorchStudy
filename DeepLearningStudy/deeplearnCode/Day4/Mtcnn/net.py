from torch import nn
import torch


class PNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 5, 1))


if __name__ == '__main__':
    net = PNet()
    x = torch.randn(2, 3, 12, 12)
    print(net(x).shape)
