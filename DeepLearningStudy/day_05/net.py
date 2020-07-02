from torch import nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, padding=1),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(10, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.outPutconv1 = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid()
        )

        self.outPutconv2 = nn.Conv2d(32, 4, 1, 1)

        self.outPutconv3 = nn.Conv2d(32, 10, 1, 1)

    def forward(self, x):
        h = self.sequential(x)
        # 置信度
        cond = self.outPutconv1(h)
        # 标注框偏移
        boxOffSet = self.outPutconv2(h)
        # 五官偏移量
        ldMoffSet = self.outPutconv3(h)
        return cond, boxOffSet, ldMoffSet


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, padding=1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.inputLayer = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.ReLU()
        )

        self.outPutconv1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.outPutconv2 = nn.Linear(128, 4)

        self.outPutconv3 = nn.Linear(128, 10)

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1, 3 * 3 * 64)
        h = self.inputLayer(h)
        # 置信度
        cond = self.outPutconv1(h)
        # 标注框偏移量
        boxOffSet = self.outPutconv2(h)
        # 五官偏移量
        ldMoffSet = self.outPutconv3(h)
        return cond, boxOffSet, ldMoffSet


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()

        )

        self.inputLayer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
        )

        self.outPutconv1 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.outPutconv2 = nn.Linear(256, 4)

        self.outPutconv3 = nn.Linear(256, 10)

    def forward(self, x):
        h = self.sequential(x)
        h = h.reshape(-1, 3 * 3 * 128)
        h = self.inputLayer(h)
        # 置信度
        cond = self.outPutconv1(h)
        # 标注框偏移量
        boxOffSet = self.outPutconv2(h)
        # 五官偏移量
        ldMoffSet = self.outPutconv3(h)
        return cond, boxOffSet, ldMoffSet


if __name__ == '__main__':
    # pnet = PNet()
    # x = torch.randn(1, 3, 12, 12)
    # cond, boxof, ladmof = pnet(x)
    # print(cond.shape, boxof.shape, ladmof.shape)

    # rnet = RNet()
    # x = torch.randn(1, 3, 24, 24)
    # cond, boxof, ladmof = rnet(x)
    # print(cond.shape, boxof.shape, ladmof.shape)

    onet = ONet()
    x = torch.randn(3, 3, 48, 48)
    cond, boxof, ladmof = onet(x)
    con1 = torch.tensor([[0.67],
                         [0.25],
                         [0.63]])
    pre = torch.tensor([[-0.0489, 0.0496, 0.0020, 0.0042],
                        [-0.0469, 0.0537, 0.0027, -0.0024],
                        [-0.0507, 0.0466, -0.0010, 0.0018]])

    con1.extend(pre)
    print(con1)
    mask = con1[:, ] > 0.65
    mask1 = 0.65 >= con1[:, ] > 0.4
    # condm = torch.nonzero(mask)[:, 0]
    condm1 = torch.nonzero(mask)[:, 0]
    print(condm1)
    # con1[condm] = 1
    # print(con1)
    # print(cond, boxof.shape, ladmof.shape)
    # print(boxof, boxof[:, ]-pre[])
