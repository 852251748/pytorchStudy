import torch, os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

tf = transforms.Compose([
    transforms.ToTensor()
])

import cv2 as cv


class MINISTDataset(Dataset):
    def __init__(self, root, is_train=True):
        self.dataset = []
        sub_dir = "TRAIN" if is_train else "TEST"

        for tag in os.listdir(f"{root}/{sub_dir}"):
            for filename in os.listdir(f"{root}/{sub_dir}/{tag}"):
                path = f"{root}/{sub_dir}/{tag}/{filename}"
                self.dataset.append([path, tag])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = Image.open(data[0])
        img = tf(img)

        tag = torch.zeros(10)
        tag[int(data[1])] = 1
        return img.reshape(-1), tag


if __name__ == '__main__':
    dataset = MINISTDataset("../../../day_01\data\MNIST_IMG", True)
    img, tag = dataset[40000]

    print(img.shape, tag)
