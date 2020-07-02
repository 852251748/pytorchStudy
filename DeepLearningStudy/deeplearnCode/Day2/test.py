from torchvision import datasets, transforms
import torch
data = datasets.CIFAR10("../../day_02\data", train=True, transform=transforms.ToTensor(), download=False)
print(data[0][0].shape)

