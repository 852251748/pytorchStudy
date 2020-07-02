import torch


class CenterLoss(torch.nn.Module):
    def __init__(self, class_num, feature_dim, lamda):
        super().__init__()
        self.class_num = class_num
        self.lamda = lamda
        self.center = torch.nn.Parameter(torch.randn(class_num, feature_dim))

    def forward(self, tag, feature):
        c = self.center[tag]
        _n = torch.histc(tag.float(), self.class_num, min=0, max=self.class_num - 1)
        n = _n[tag]
        d = (((feature - c) ** 2).sum(1)) ** 0.5
        loss = (d / n).sum() / 2.0 * self.lamda
        return loss


if __name__ == '__main__':
    tag = torch.tensor([1, 2, 3, 3])
    # tag = torch.tensor([0,0,1])
    feature = torch.tensor([[0.6, 0.6], [0.9, 0.9], [0.2, 0.2], [0.7, 0.7]])
    center = torch.tensor(
        [[0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3],
         [0.6, 0.6]])
    loss = CenterLoss(10, 2)
    print(loss(tag, feature))
