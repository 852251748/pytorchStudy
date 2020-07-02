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


class Block11(nn.Module):
    def __init__(self, ichannel, t, c, n, s, j):
        super().__init__()
        self.lastT = True if n - j == 1 else False
        pC = ichannel * t
        oS = s if self.lastT else 1
        oC = c if self.lastT else ichannel
        self.sequential = nn.Sequential(
            nn.Conv2d(ichannel, pC, 1, 1, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU6(),
            nn.Conv2d(pC, pC, 3, oS, padding=1, groups=pC, bias=False),
            nn.BatchNorm2d(pC),
            nn.ReLU6(),
            nn.Conv2d(pC, oC, 1, 1, bias=False),
            nn.BatchNorm2d(oC),
        )
        self.relu = nn.ReLU6()

    def forward(self, x):
        h = self.sequential(x)
        if not self.lastT:
            h = h + x
        return self.relu(h)


class MobileNet_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inputLayer = nn.Sequential(
            nn.Conv2d(3, config[0][1], 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(config[0][1]),
            nn.ReLU6()
        )
        layerPro = []
        ichannel = config[0][1]
        for t, c, n, s in config[1:]:
            for j in range(n):
                layerPro.append(Block11(ichannel, t, c, n, s, j))
            ichannel = c

        self.hideLayer = nn.Sequential(*layerPro)

        self.outputLayer = nn.Sequential(
            nn.Conv2d(ichannel, 1280, 1, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(7),
            nn.Conv2d(1280, 10, 1)
        )

    def forward(self, x):
        h = self.inputLayer(x)
        # print(h.shape)
        h = self.hideLayer(h)
        h = self.outputLayer(h)
        h = h.reshape(-1,10)
        return h


if __name__ == '__main__':
    net = MobileNet_v3(config)
    x = torch.randn(2, 3, 224, 224)
    print(net(x).shape)
