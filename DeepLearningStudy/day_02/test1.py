import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import nn


# net = models.resnet18()
# print(net)
# # m = nn.ZeroPad2d(16)
# # x = torch.randn(1, 3, 32, 32)
# # print(m(x).shape)
train_data = datasets.CIFAR10("./data", True, transform=transforms.ToTensor(), download=True)
print(train_data[0][0].shape)

