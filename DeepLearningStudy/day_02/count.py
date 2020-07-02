import torch
import thop
from torch import nn

if __name__ == '__main__':
    conv = nn.Conv2d(3, 16, 3, 1)
    x = torch.randn(1, 3, 16, 16)
    flop, parma = thop.profile(conv, (x,))
    flop, parma = thop.clever_format((flop, parma), "%.3f")
    print(flop, parma)
