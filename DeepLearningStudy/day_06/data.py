from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image

tf = transforms.ToTensor()


class MyData(Dataset):
    def __init__(self, root):
        self.dataPath = root
        self.dataset = os.listdir(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fn = self.dataset[index]
        strs = fn.split(".")[0]

        img = tf(Image.open(f"{self.dataPath}/{fn}"))
        # lable = np.array([int(x) for x in strs])
        lable = torch.zeros(4, 10)
        for i, x in enumerate(strs):
            lable[i][int(x)] = 1
        return img, lable


if __name__ == '__main__':
    mydataset = MyData("./code")
    print(mydataset[0])
