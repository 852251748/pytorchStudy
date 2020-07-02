import torch
from torch import nn

config = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, s=1, padding=0):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, s, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


class Block(nn.Module):
    def __init__(self, in_channel, t, _c, _s):
        super().__init__()
        self.Flag = in_channel == _c
        self.seq = nn.Sequential(
            ConvBNReLU(in_channel, in_channel * t, 1),
            ConvBNReLU(in_channel * t, in_channel * t, s=_s, padding=1),
            nn.Conv2d(in_channel * t, _c, 1, 1),
            nn.BatchNorm2d(_c)
        )

    def forward(self, x):
        h = self.seq(x)
        if self.Flag != 1:
            return h
        return h + x


class MobileNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        hide_layer = []
        in_channel = 32
        for t, c, n, s in config:
            for i in range(n):
                # 每一层的最后一个批次使用config中的步长和通道，其他层都使用1的步长和输出通道
                _s = s if n - i == 1 else 1
                _c = c if n - i == 1 else in_channel
                hide_layer.append(Block(in_channel, t, _c, _s))
                in_channel = _c

        self.hide_layer = nn.Sequential(*hide_layer)

        self.output_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 3, 1, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.AvgPool2d(7),
            nn.Conv2d(1280, 10, 1, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hide_layer(h)
        h = self.output_layer(h)
        return h


if __name__ == '__main__':
    net = MobileNet(config)
    # print(net)
    x = torch.randn(2, 3, 224, 224)
    print(net(x).shape)
