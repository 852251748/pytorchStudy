from torch import nn
import torch


class DNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
        )


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(  # 输入noise形状为[batch,128,1,1]
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.sequential(noise)


class GANNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet = DNet()
        self.gnet = GNet()
        self.lossFun = nn.BCEWithLogitsLoss()

    def forward(self, noiseG):
        y = self.gnet(noiseG)
        return y

    def GetDLoss(self, noiseD, realImg):
        realY = self.dnet(realImg).reshape(-1)
        fakeImg = self.gnet(noiseD)
        fakeY = self.dnet(fakeImg).reshape(-1)

        realTag = torch.ones(realImg.size(0)).cuda()
        fakeTag = torch.zeros(fakeImg.size(0)).cuda()

        lossD1 = self.lossFun(realY, realTag)
        lossD2 = self.lossFun(fakeY, fakeTag)

        return lossD1 + lossD2

    def GetGLoss(self, noiseG):
        fakeImg = self.gnet(noiseG)
        fakeY = self.dnet(fakeImg).reshape(-1)

        realTag = torch.ones(fakeImg.size(0)).cuda()
        lossG = self.lossFun(fakeY, realTag)
        return lossG


if __name__ == '__main__':
    # dnet = DNet()
    # x = torch.randn(1, 3, 96, 96)
    # y = dnet(x)
    # print(y.shape)

    gnet = GNet()
    noise = torch.randn(2, 128, 1, 1)
    y = gnet(noise)
    print(y.shape)
