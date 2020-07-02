from torch import nn
import torch
import numpy as np


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(10, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 5, 1, 1)
        )

    def forward(self, x):
        h = self.sequential(x)
        return h


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.ReLU()
        )

        self.inputLayer = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1, 3 * 3 * 64)

        return self.inputLayer(h)


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 2, 1),
            nn.ReLU()

        )

        self.inputLayer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1, 3 * 3 * 128)

        return self.inputLayer(h)


if __name__ == '__main__':
    # pnet = PNet()
    # x = torch.randn(1, 3, 12, 12)
    # h = pnet(x)
    #
    # print(h.shape)

    # rnet = RNet()
    # x = torch.randn(2, 3, 24, 24)
    # h = rnet(x)
    #
    # print(h.shape)

    onet = ONet()
    x = torch.randn(1, 3, 48, 48)
    h = onet(x)

    print(h.shape)
