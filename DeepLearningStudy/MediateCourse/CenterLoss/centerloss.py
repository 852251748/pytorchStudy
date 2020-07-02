import torch


class CenterLoss(torch.nn.Module):
    def __init__(self, feature_dim, class_num):
        super().__init__()
        self.class_num = class_num
        self.center = torch.nn.Parameter(torch.randn(class_num, feature_dim))

    def forward(self, tags, feature):
        c = self.center[tags]

        _n = torch.histc(tags.float(), self.class_num, min=0, max=self.class_num - 1)

        n = _n[tags]

        d = ((feature - c) ** 2).sum(dim=1) ** 0.5
        loss = (d / n).sum()
        return loss
