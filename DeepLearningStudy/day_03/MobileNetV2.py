import torch
from torch import nn

config = [[-1, 32, 1, 2],
          [1, 16, 1, 1],
          [6, 24, 2, 2],
          [6, 32, 3, 2],
          [6, 64, 4, 2],
          [6, 96, 3, 1],
          [6, 160, 3, 2],
          [6, 320, 1, 1]]


class Block(nn.Module):
    def __init__(self, iChannel, t, oChannel, num, stride, j):
        super().__init__()
        self.lastT = True if num - j == 1 else False
        pC = iChannel * t
        pS = stride if self.lastT else 1
        oC = oChannel if self.lastT else iChannel
        self.layer = nn.Sequential(
            nn.Conv2d(iChannel, pC, 1, 1, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU(),
            nn.Conv2d(pC, pC, 3, pS, padding=1, groups=pC, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU(),
            nn.Conv2d(pC, oC, 1, 1, bias=False),
        )

    def forward(self, x):
        h = self.layer(x)
        if not self.lastT:
            h = h + x
        return h


class MobileNet(nn.Module):
    def __init__(self, iChannel, config):
        super().__init__()
        self.inputLayer = nn.Sequential(
            nn.Conv2d(iChannel, config[0][1], 3, config[0][3], padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        bottleNeck = []
        iChannel = config[0][1]
        for i, (t, c, n, s) in enumerate(config[1:]):
            for j in range(n):
                bottleNeck.append(Block(iChannel, t, c, n, s, j))
            iChannel = c

        self.hideLayer = nn.Sequential(*bottleNeck)

        self.outputLayer = nn.Sequential(
            nn.Conv2d(iChannel, 1280, 1, 1),
            nn.AvgPool2d(7),
            nn.Conv2d(1280, 10, 1)
        )

    def forward(self, x):
        h = self.inputLayer(x)
        h = self.hideLayer(h)
        h = self.outputLayer(h)
        h = h.reshape(-1, 10)
        return h


if __name__ == '__main__':
    net = MobileNet(3, config)
    a = torch.randn(1, 3, 224, 224)
    print(net(a).shape)
