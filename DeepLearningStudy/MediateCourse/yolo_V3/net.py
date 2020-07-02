from torch import nn
import torch
from torch.nn import functional as F
from MediateCourse.yolo_V3.FRn import FilterResponseNormalization


# 定义卷积层
class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            # FilterResponseNormalization(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sequential(x)


# 定义残差块
class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sequential = nn.Sequential(
            Convolutional(in_channels, in_channels // 2, 1, 1, 0),
            Convolutional(in_channels // 2, in_channels, 3, 1, padding=1),
        )

    def forward(self, x):
        return x + self.sequential(x)


# 定义下采样
class DownsamplingLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.sequential = nn.Sequential(
            Convolutional(in_channel, out_channel, 3, 2, 1),
        )

    def forward(self, x):
        return self.sequential(x)


# 定义上采样
class UpsamplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


# 定义卷积块
class ConvolutionalSet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.sequential = nn.Sequential(
            Convolutional(in_channel, out_channel, 1, 1, 0),
            Convolutional(out_channel, in_channel, 3, 1, 1),
            Convolutional(in_channel, out_channel, 1, 1, 0),
            Convolutional(out_channel, in_channel, 3, 1, 1),
            Convolutional(in_channel, out_channel, 1, 1, 0),
        )

    def forward(self, x):
        return self.sequential(x)


# 定义主网络
class MainNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk_52 = nn.Sequential(
            Convolutional(3, 32, 3, 1, 1),
            DownsamplingLayer(32, 64),
            Residual(64),
            DownsamplingLayer(64, 128),
            Residual(128),
            Residual(128),
            DownsamplingLayer(128, 256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
        )

        self.trunk_26 = nn.Sequential(
            DownsamplingLayer(256, 512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
        )

        self.trunk_13 = nn.Sequential(
            DownsamplingLayer(512, 1024),
            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024),
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024, 512),
        )

        self.detetion_13 = nn.Sequential(
            Convolutional(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 21, 1, 1, 0)
        )

        self.upsample1 = nn.Sequential(
            Convolutional(512, 256, 1, 1, 0),
            UpsamplingLayer()
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalSet(768, 256),
        )

        self.detetion_26 = nn.Sequential(
            Convolutional(256, 512, 3, 1, 1),
            nn.Conv2d(512, 21, 1, 1, 0)
        )

        self.upsample2 = nn.Sequential(
            Convolutional(256, 128, 1, 1, 0),
            UpsamplingLayer()
        )

        self.convset_52 = nn.Sequential(
            ConvolutionalSet(384, 128),
        )

        self.detetion_52 = nn.Sequential(
            Convolutional(128, 256, 3, 1, 1),
            nn.Conv2d(256, 21, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up26_out = self.upsample1(convset_out_13)
        cat26_output = torch.cat((up26_out, h_26), dim=1)
        convset_out_26 = self.convset_26(cat26_output)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up52_out = self.upsample2(convset_out_26)
        cat52_output = torch.cat((up52_out, h_52), dim=1)
        convset_out_52 = self.convset_52(cat52_output)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    # net = Residual(4)
    # net = DownsamplingLayer(4, 8)
    # net = UpsamplingLayer()
    net = MainNet()
    input = torch.randn(1, 3, 416, 416)
    y13, y26, y52 = net(input)
    print(y13.shape, y26.shape, y52.shape)
