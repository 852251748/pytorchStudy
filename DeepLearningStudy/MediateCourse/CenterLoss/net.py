from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequaltion = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.liner_layer = nn.Linear(128 * 4 * 4, 2)
        self.output = nn.Linear(2, 10)

    def forward(self, x):
        output = self.sequaltion(x)
        output = output.reshape(-1, 128 * 4 * 4)
        feature = self.liner_layer(output)
        y = self.output(feature)
        return feature, y


if __name__ == '__main__':
    a = torch.randn(2, 1, 28, 28)
    net = Net()
    feature, y = net(a)
    print(feature.shape, y.shape)
