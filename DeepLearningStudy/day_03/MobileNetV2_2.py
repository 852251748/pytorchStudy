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


class Block1(nn.Module):
    def __init__(self, iChannel, t, c, n, s, j):
        super().__init__()
        self.lastT = True if n - j == 1 else False

        pC = iChannel * t
        sP = s if self.lastT else 1
        oC = c if self.lastT else iChannel
        self.layer = nn.Sequential(
            nn.Conv2d(iChannel, pC, 1, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU6(),
            nn.Conv2d(pC, pC, 3, sP, padding=1, groups=pC, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU6(),
            nn.Conv2d(pC, oC, 1, bias=False),
            nn.BatchNorm2d(oC)
        )

    def forward(self, x):
        h = self.layer(x)
        if not self.lastT:
            h = h + x
        return h


class MobileNet1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.inputLayer = nn.Sequential(
            nn.Conv2d(3, config[0][1], 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(config[0][1]),
            nn.ReLU6()
        )

        layer = []
        iPutChannel = config[0][1]
        for i, (t, c, n, s) in enumerate(config[1:]):
            for j in range(n):
                layer.append(Block1(iPutChannel, t, c, n, s, j))
            iPutChannel = c

        self.hideLayer = nn.Sequential(*layer)

        self.outPut = nn.Sequential(
            nn.Conv2d(iPutChannel, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(2),
            nn.Conv2d(1280, 10, 1, 1, bias=False)
        )

        def forward(self, x):
            h = self.inputLayer(x)
            # print("1", h.shape)
            h = self.hideLayer(h)
            # print("2", h.shape)
            h = self.outPut(h)
            # print("3", h.shape)
            h = h.reshape(-1, 10)
            return h

    if __name__ == '__main__':
        net = MobileNet1(config)
        x = torch.randn(3, 3, 32, 32)
        print(net(x).shape)
