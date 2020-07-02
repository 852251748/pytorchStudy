import torch
from torch import nn
from torchvision import models
from torchvision import datasets, transforms


class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU()
        )

        self.liner = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        h = self.seq(x)
        h = h.reshape(-1, 128 * 4 * 4)

        return self.liner(h)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class NetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU()
        )

        self.liner = nn.Linear(128 * 4 * 4, 10)

        self.apply(weight_init)

    def forward(self, x):
        h = self.seq(x)
        h = h.reshape(-1, 128 * 4 * 4)

        return self.liner(h)


class BasicBlockT(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x) + x


# 训练8轮左右精度到85%左右
class NetV2T(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(3, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BasicBlockT(32),
            BasicBlockT(32),
            # -----------------------------------------------
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # -----------------------------------------------
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockT(64),
            BasicBlockT(64),
            # -----------------------------------------------
            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # -----------------------------------------------
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BasicBlockT(128),
            BasicBlockT(128),
            # -----------------------------------------------
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # -----------------------------------------------
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            BasicBlockT(256),
            BasicBlockT(256),
            # -----------------------------------------------
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            # -----------------------------------------------
        )

        self.liner = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.BatchNorm1d(4096),  # 输入批次要大于1,BatchNorm1d是计算多个批次的平均值的
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),  # 输入批次要大于1,BatchNorm1d是计算多个批次的平均值的
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

        # self.apply(weight_init)

    def forward(self, x):
        h = self.seq(x)
        h = h.reshape(-1, 256 * 4 * 4)
        return self.liner(h)


class NetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64),
        )

        self.layer2 = nn.Sequential(
            Upsample(64, 128),
            BasicBlock(128),
            BasicBlock(128)
        )

        self.layer3 = nn.Sequential(
            Upsample(128, 256),
            BasicBlock(256),
            BasicBlock(256)
        )

        self.layer4 = nn.Sequential(
            Upsample(256, 512),
            BasicBlock(512),
            BasicBlock(512)
        )

        self.liner = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1000),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        h = self.seq(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer4(h)
        h = h.reshape(-1, 256 * 4 * 4)
        return self.liner(h)


class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.seq(x) + x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3, 2, 1)
        )

    def forward(self, x):
        return self.seq(x)


class NetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()

        self.liner = nn.Linear(1000, 10)

    def forward(self, x):
        h = self.resnet(x)

        return self.liner(h)


config = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


class Bootleneck(nn.Module):
    def __init__(self, in_channel, t, c, s):
        super().__init__()
        self.Flag = in_channel == c
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * t, 1, 1),
            nn.BatchNorm2d(in_channel * t),
            nn.ReLU(),
            nn.Conv2d(in_channel * t, in_channel * t, 3, s, padding=1),
            nn.BatchNorm2d(in_channel * t),
            nn.ReLU(),
            nn.Conv2d(in_channel * t, c, 1, 1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        h = self.seq(x)
        if self.Flag:
            return h + x
        return h


class MobileNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        block = []
        in_channel = 32
        for t, c, n, s in config:
            for i in range(n):
                _s = s if n - i == 1 else 1
                _c = c if n - i == 1 else in_channel
                block.append(Bootleneck(in_channel, t, _c, _s))
                in_channel = _c

        self.hide_layer = nn.Sequential(*block)

        self.output_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.AvgPool2d(7),
            nn.Conv2d(1280, 10, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hide_layer(h)
        h = self.output_layer(h)
        return h

# VGG19
class NetV5(nn.Module):
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


if __name__ == '__main__':
    # train_dataset = datasets.CIFAR10("D:\Alldata", train=True, transform=transforms.ToTensor(),
    #                                  download=False)
    # print(train_dataset[0][0].shape)
    net = NetV2T()
    # print(net)
    x = torch.rand(3, 3, 32, 32)
    y = net(x)
    print(y.shape)

    # net1 = models.resnet18()
    # print(net1)
