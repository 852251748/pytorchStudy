import torch
from torch import nn


class CE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ys, tags):
        h = -tags * torch.log(ys)
        # h = -ys * torch.log(tags)
        print("h = ", h)
        return torch.mean(h)


if __name__ == '__main__':
    ce = CE()
    y = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    print("y:", y)
    tags = torch.tensor([[1, 0], [0, 1]])
    print(ce.forward(y, tags))
