from torch import nn
import torch


class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.squential = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        h = self.squential(x)
        return nn.functional.softmax(h, dim=1)


if __name__ == '__main__':
    net = MlpNet()
    x = torch.randn((2, 784))

    y = net(x)
    print(torch.sum(y, dim=1))
