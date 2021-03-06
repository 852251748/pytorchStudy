from torch import nn
import torch

class PNet1(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 15, 1)
        )


class PNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 10, 3, 1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 15, 1)
        )


class PNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 15, 1, 1)
        )
        self.feature_layer = nn.Conv2d(1, 2, 1, 1)
        self.output_layer = nn.Conv2d(16, 15, 1, 1)

    def forward(self, x):
        h = self.input_layer(x)
        feature = self.feature_layer(h[:, 0][:, None, ...])
        h = torch.cat([h[:, 1:], feature], dim=1)
        return self.output_layer(h), feature


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),
            nn.ReLU(),
            nn.Conv2d(28, 28, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 48, 3, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 2, 1),
            nn.ReLU()
        )

        self.liner_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h.reshape(-1, 64 * 3 * 3)
        return self.liner_layer(h)


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, 1),
            nn.ReLU()
        )

        self.liner_layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h.reshape(-1, 128 * 3 * 3)
        return self.liner_layer(h)


if __name__ == '__main__':
    net = PNet2()
    x = torch.randn(2, 3, 12, 12)
    output, feature = net(x)
    print(output.shape, feature.shape)
