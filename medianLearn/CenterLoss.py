import torch


class CenterLoss(torch.nn.Module):
    def __init__(self, class_num, feature_dim):
        super().__init__()
        self.center = torch.nn.Parameter(torch.tensor(class_num, feature_dim))

    def forward(self, tag,feature):
        c = self.center[tag]
        _n = torch.histc(tag.float(), 2)
        n = _n[tag]
        d = (((feature - c) ** 2).sum(1)) ** 0.5
        loss = (d / n).sum()
        return loss