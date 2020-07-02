from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

tf = transforms.Compose([
    transforms.ToTensor()
])


class Mydataset(Dataset):
    def __init__(self, root, imagesize, isTest=False):
        self.imgsize = imagesize
        self.dataset = []
        self.dataDir = f"{root}/{imagesize}"

        dirList = [f"{self.dataDir}/positive.txt", f"{self.dataDir}/negative.txt", f"{self.dataDir}/part.txt"]

        # 测试数据集和训练数据集比例 1:9
        with open(f"{self.dataDir}/positive.txt") as f1:
            self.dataset.extend(f1.readlines())
        with open(f"{self.dataDir}/negative.txt") as f2:
            self.dataset.extend(f2.readlines())
        with open(f"{self.dataDir}/part.txt") as f3:
            self.dataset.extend(f3.readlines())
        # for i in range(len(dirList)):
        #     with open(dirList[i])as f:
        #         fileList = f.readlines()
        #         if isTest:
        #             fileList = fileList[int(len(fileList) * 0.9):]
        #         else:
        #             fileList = fileList[:int(len(fileList) * 0.9)]
        #         self.dataset.extend(fileList)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        data = data.strip().split()

        if int(data[1]) == 1:
            img = tf(Image.open(f"{self.dataDir}/positive/{data[0].strip()}"))
        elif int(data[1]) == 0:
            img = tf(Image.open(f"{self.dataDir}/negative/{data[0].strip()}"))
        else:
            img = tf(Image.open(f"{self.dataDir}/part/{data[0].strip()}"))

        # 返回图片数据，置信度，偏移量
        cond = np.array([float(data[1])], dtype=np.float32)
        boxOffSet = np.array([float(data[2]), float(data[3]), float(data[4]), float(data[5])], dtype=np.float32)
        ldMkOffSet = np.array(
            [float(data[6]), float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]),
             float(data[12]), float(data[13]), float(data[14]), float(data[15])], dtype=np.float32)
        # if self.imgsize == 12:
        #     return img, cond, np.hstack((boxOffSet, ldMkOffSet), )
        return img, cond, boxOffSet, ldMkOffSet


if __name__ == '__main__':
    mydataset = Mydataset("D:/pycharm_workspace/mtcnn_data", 12)
    print(mydataset[0])
