import cv2 as cv
from torch.utils.data import Dataset
import os
import numpy as np


class Mydataset(Dataset):
    def __init__(self, root, IS_TRAIN):
        self.root = root

        if IS_TRAIN:
            imgPath = f"{root}/MNIST_IMG/TRAIN"
        else:
            imgPath = f"{root}/MNIST_IMG/TEST"

        self.dataset = []

        for i in os.listdir(imgPath):
            picNames = os.listdir(f"{imgPath}/{i}")
            for picName in picNames:
                picpath = f"{imgPath}/{i}/{picName}"
                self.dataset.append((picpath, i))

        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        tag = data[1]
        imgPath = data[0]

        img = cv.imread(imgPath, 0)
        img = img[None, ...]
        img = img / 255
        return np.float32(img), int(tag)


if __name__ == '__main__':
    mydataset = Mydataset("../../work/data", True)
    print(len(mydataset))
    print(mydataset[0])
