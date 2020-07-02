from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np


class MyFaceData(Dataset):
    def __init__(self, path):
        self.dataset = os.listdir(path)
        self.dataPath = path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fn = self.dataset[index]
        imgData = cv.imread(f"{self.dataPath}/{fn}")
        imgData = cv.cvtColor(imgData, cv.COLOR_BGR2RGB)
        imgData = np.transpose(imgData, [2, 0, 1])
        imgData = ((imgData / 255. - 0.5) * 2).astype(np.float32)

        return imgData


if __name__ == '__main__':
    mydata = MyFaceData(r"E:\BaiduNetdiskDownload\DeepLearingDownLoad\2020-04-24GAN\newCode\Cartoon_faces\faces")
    # mydata = MyFaceData(r"E:\BaiduNetdiskDownload\DeepLearingDownLoad\2020-04-24GAN\newCode\Cartoon_faces")

    mydata[1]
