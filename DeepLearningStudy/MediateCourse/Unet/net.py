import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.sequential(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.Layer = nn.Conv2d(channel, channel // 2, 3, 1, 1)

    def forward(self, x, r):
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.Layer(h)
        return torch.cat((h, r), dim=1)


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = ConvLayer(3, 64)
        self.D1 = DownSample(64)
        self.C2 = ConvLayer(64, 128)
        self.D2 = DownSample(128)
        self.C3 = ConvLayer(128, 256)
        self.D3 = DownSample(256)
        self.C4 = ConvLayer(256, 512)
        self.D4 = DownSample(512)
        self.C5 = ConvLayer(512, 1024)

        self.U1 = UpSample(1024)
        self.C6 = ConvLayer(1024, 512)
        self.U2 = UpSample(512)
        self.C7 = ConvLayer(512, 256)
        self.U3 = UpSample(256)
        self.C8 = ConvLayer(256, 128)
        self.U4 = UpSample(128)
        self.C9 = ConvLayer(128, 64)

        self.end = nn.Conv2d(64, 3, 3, 1, 1)
        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        C1_out = self.C1(x)
        C2_out = self.C2(self.D1(C1_out))
        C3_out = self.C3(self.D2(C2_out))
        C4_out = self.C4(self.D3(C3_out))
        C5_out = self.C5(self.D4(C4_out))
        C6_out = self.C6(self.U1(C5_out, C4_out))
        C7_out = self.C7(self.U2(C6_out, C3_out))
        C8_out = self.C8(self.U3(C7_out, C2_out))
        C9_out = self.C9(self.U4(C8_out, C1_out))
        out = self.end(C9_out)
        return self.active(out)


if __name__ == '__main__':
    # net = ConvLayer(3, 64)
    # net = DownSample(3, 64)
    net = MainNet()
    # net = UpSample(64)
    a = torch.randn(2, 3, 256, 256)

    print(net(a).shape)
