import torch

feature = torch.tensor([[0.3, 0.3], [0.5, 0.5], [0.8, 0.8]])
tags = torch.tensor([0, 1, 1])
center = torch.tensor([[0.3, 0.3], [0.6, 0.6]])

c = center[tags]

_n = torch.histc(tags.float(), 2, min=0, max=1)

n = _n[tags]

d = ((feature - c) ** 2).sum(dim=1) ** 0.5
loss = (d / n).sum()
print(loss)
