from torch import nn
import torch

DEVICE = "cuda:0"


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128 * 30 * 8, 128, 2, batch_first=True)
        self.outPut = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        x = x.reshape(-1, 128 * 30 * 8)
        x = x[:, None, :].repeat(1, 4, 1)
        h0 = torch.zeros(2 * 1, x.size(0), 128).to(DEVICE)

        output, hn = self.rnn(x, h0)
        return self.outPut(output)


class Cnn2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


if __name__ == '__main__':
    # x = torch.randn(1, 3, 60, 240)
    # encode = Encoder()
    # y = encode(x)
    # print(y.shape)
    # x = torch.randn(1, 256, 7, 30)
    # decode = Decoder()
    # y = decode(x)
    # # print(y.shape)
    x = torch.randn(2, 3, 60, 240)
    net = Cnn2Seq()
    y = net(x)
    print(y.shape)
