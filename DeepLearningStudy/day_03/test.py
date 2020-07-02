import torch
from torch import nn

config = [
    [-1, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


class Block(nn.Module):

    def __init__(self, p_c, i, t, c, n, s):
        super().__init__()

        # 上课的时候这里讲得有点问题
        # 每个重复的最后一次负责降采样和通道
        # 讲课的时候说的是重复的第一次负责
        # 所以正确的是i==n-1的时候进行操作

        self.i = i
        self.n = n

        _s = s if i == n - 1 else 1  # 判断是否是最后一次重复，最后一次重复步长为2
        _c = c if i == n - 1 else p_c  # 判断是否是最后一次重复，最后一次重复负责通道变换为下层的输出

        _p_c = p_c * t  # 输入通道通道扩增倍数

        self.layer = nn.Sequential(
            nn.Conv2d(p_c, _p_c, 1, _s, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _p_c, 3, 1, padding=1, groups=_p_c, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self, x):
        if self.i == self.n - 1:
            return self.layer(x)
        else:
            return self.layer(x) + x


class MobilenetV2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = []
        p_c = config[0][1]
        for t, c, n, s in config[1:]:
            for i in range(n):
                self.blocks.append(Block(p_c, i, t, c, n, s))
            p_c = c
        self.hidden_layer = nn.Sequential(*self.blocks)

        self.ouput_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(7, 1),
            nn.Conv2d(1280, 10, 1, 1, bias=False)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hidden_layer(h)
        h = self.ouput_layer(h)
        h = h.reshape(-1, 10)  # 这里直接做个形状变换
        return h


if __name__ == '__main__':
    net = MobilenetV2(config)
    y = net(torch.randn(1, 3, 448, 448))
    print(y.shape)
