import torch
from torch import nn
from torchvision import models


class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.squential = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.ReLU()
        )

        self.liner_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 10),
        )

    def forward(self, xs):
        h = self.squential(xs)
        h = h.reshape(-1, 64 * 8 * 8)
        return self.liner_layer(h)


# 权重初始化
def weight_init(m):
    if (isinstance(m, nn.Conv2d) | isinstance(m, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    # if (isinstance(m, nn.Conv2d)):
    #     nn.init.kaiming_normal_(m.weight)
    #     nn.init.zeros_(m.bias)
    # elif(isinstance(m,nn.Linear)):
    #     nn.init.kaiming_normal_(m.weight)
    #     nn.init.zeros_(m.bias)


class NetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.squential = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.liner_layer = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(256 * 2 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

        self.apply(weight_init)

    def forward(self, xs):
        h = self.squential(xs)
        h = h.reshape(-1, 256 * 2 * 2)
        return self.liner_layer(h)


# VGG19
class NetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.squential = nn.Sequential(
            # nn.Dropout2d(0.2),
            # nn.ZeroPad2d(16),
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 输出变成 64*32*32

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 输出变成 128*16*16

            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # 输出变成 256*8*8

            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  # 输出变成 512*4*4

            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  # 输出变成 512*2*2
        )

        self.liner_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

        self.apply(weight_init)

    def forward(self, xs):
        h = self.squential(xs)
        # return h
        h = h.reshape(-1, 512 * 2 * 2)
        return self.liner_layer(h)


# 残差网络
class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, downsampling=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i_channel, o_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(o_channel),
            nn.ReLU(),
            nn.Conv2d(o_channel, o_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(o_channel),
            # nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.down_sampling = downsampling

    def forward(self, xs):
        residual = xs
        h = self.block(xs)
        # print(h.shape)
        if self.down_sampling:
            residual = self.down_sampling(xs)
            # print("sx.shape", xs.shape)

        return self.relu(h + residual)


class NetV4(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(3, 2)
        )

        self.conv2 = self.make_retblocks(32, 32)
        self.conv3 = self.make_retblocks(32, 64, 2)
        self.conv4 = self.make_retblocks(64, 128, 2)
        self.conv5 = self.make_retblocks(128, 256, 2)
        self.conv6 = self.make_retblocks(256, 512, 2)
        self.avgpool = nn.AvgPool2d(2)
        self.liner = nn.Sequential(
            nn.Linear(512 * 1 * 1, 10)
        )

    def make_retblocks(self, i_channel, o_channel, stride=1):
        downsamp = None
        if stride != 1:
            downsamp = nn.Sequential(
                nn.Conv2d(i_channel, o_channel, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(o_channel),
            )

        layer = [Residual_Block(i_channel, o_channel, stride, downsampling=downsamp),
                 Residual_Block(o_channel, o_channel)]

        return nn.Sequential(*layer)

    def forward(self, xs):
        h = self.conv1(xs)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.avgpool(h)
        return self.liner(h.reshape(-1, 512 * 1 * 1))


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
        self.convX = nn.Conv2d(iChannel, oC, 3, sP, padding=1, bias=False) if self.lastT else None
        self.relu = nn.ReLU6()

    def forward(self, x):
        h = self.layer(x)
        if self.convX:
            x = self.convX(x)

        # if not self.lastT:
        #     h = h + x

        return self.relu(h + x)


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
    # net = NetV4()
    x = torch.randn(1, 3, 32, 32)
    print(net(x).shape)
