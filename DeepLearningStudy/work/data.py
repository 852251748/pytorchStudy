# 数据加载及处理
import os, cv2
from torch.utils.data import Dataset
import numpy as np


class MNISTDataset(Dataset):

    def __init__(self, datapath, IS_TRAIN):  # 加载数据保存图片的路径和标签
        self.dataset = []
        sub_dir = "TRAIN" if IS_TRAIN else "TEST"

        for tags in os.listdir(f"{datapath}/{sub_dir}"):
            img_dir = f"{datapath}/{sub_dir}/{tags}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path, tags))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):  # 加载图片信息，标签热编码
        data = self.dataset[item]
        # 读取图片
        imgdata = cv2.imread(data[0], 0)
        # # 改变形状成为一行
        imgdata = imgdata.reshape(1, 28, 28)
        # 数据归一化
        imgdata = imgdata / 255

        # # 标签Hot—code
        # tags_HotCode = np.zeros(10)
        # tags_HotCode[int(data[1])] = 1
        return np.float32(imgdata), int(data[1])


if __name__ == "__main__":
    dataset = MNISTDataset("data/MNIST_IMG", True)
    print(dataset[30000][0].shape)
    print(len(dataset))
