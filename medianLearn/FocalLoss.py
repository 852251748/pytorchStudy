import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alaph = alpha
        self.gamma = gamma

    def forward(self, x, tag):
        softmax = x[tag == 1]
        return -self.alaph * (1 - softmax) ** self.gamma * torch.log(softmax)
