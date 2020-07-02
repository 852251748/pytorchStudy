import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image

tf = transforms.Compose([
    transforms.ToTensor()
])


class Mydataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.label_path = f"{root}/SegmentationClass"
        self.img_path = f"{root}/JPEGImages"
        self.label_dataset = os.listdir(self.label_path)

    def __len__(self):
        return len(self.label_dataset)

    def __getitem__(self, item):
        data = self.label_dataset[item]
        jpgname = data[:-3] + "jpg"
        black01 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        black02 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        input = Image.open(f"{self.img_path}/{jpgname}")
        label = Image.open(f"{self.label_path}/{data}")
        w, h = input.size
        maxside = max(w, h)
        ratio = 256. / maxside
        w, h = w * ratio, h * ratio
        input = input.resize((int(w), int(h)))
        label = label.resize((int(w), int(h)))

        black01.paste(input, (0, 0, int(w), int(h)))
        black02.paste(label, (0, 0, int(w), int(h)))

        return tf(black01), tf(black02)


if __name__ == '__main__':
    dataset = Mydataset(r"D:\Alldata\VOCdevkit\VOC2012")
    input, lable = dataset[0]
    print(input.shape, lable.shape)
