import torch

tag = torch.tensor([1,2,3,3])
# tag = torch.tensor([0,0,1])
feature = torch.tensor([[0.6, 0.6], [0.9, 0.9], [0.2, 0.2], [0.7, 0.7]])
center = torch.tensor(
    [[0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3], [0.6, 0.6], [0.3, 0.3],
     [0.6, 0.6]])

c = center[tag]

_n = torch.histc(tag.float(), 10, min=0, max=9)
# print(_n)
n = _n[tag]
print(feature.shape,n.shape,c.shape)
d = (((feature - c) ** 2).sum(1)) ** 0.5
loss = (d / n).sum()
print(loss)
