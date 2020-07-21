import torch
from torch import nn


class CenterLoss(nn.Module):
    def __init__(self, class_num, feature_dim, lmada):
        super().__init__()
        self.lamada = lmada
        self.center = nn.Parameter(torch.randn(class_num, feature_dim), True)
        self.class_num = class_num

    def forward(self, tag, feature):
        c = self.center[tag.long()]
        _n = torch.histc(tag.float(), self.class_num, min=0, max=self.class_num - 1)
        n = _n[tag.long()]
        d = (((feature - c) ** 2).sum(1)) ** 0.5
        loss = (d / n).sum() / 2.0 * self.lamada
        return loss


if __name__ == '__main__':
    tag = torch.tensor([1, 2, 3, 3])
    # tag = torch.tensor([0,0,1])
    feature = torch.tensor([[0.6, 0.6], [0.9, 0.9], [0.2, 0.2], [0.7, 0.7]])
    center = torch.tensor(
        [[0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3],
         [0.6, 0.6]])
    loss = CenterLoss(10, 2, 0.2)
    print(loss(tag, feature))
