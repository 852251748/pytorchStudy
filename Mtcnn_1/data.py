from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

tf = transforms.Compose([
    transforms.ToTensor()
])


class Mydataset(Dataset):
    def __init__(self, root, imgsize):
        self.imgsize = imgsize
        self.dataPath = f"{root}/{imgsize}"

        self.dataset = []

        with open(f"{self.dataPath}/positive.txt") as f:
            self.dataset.extend(f.readlines())

        with open(f"{self.dataPath}/negative.txt") as f:
            self.dataset.extend(f.readlines())

        with open(f"{self.dataPath}/part.txt") as f:
            self.dataset.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        datas = data.strip().split()
        if int(datas[1]) == 1:
            img = Image.open(f"{self.dataPath}/positive/{datas[0]}")
        elif int(datas[1]) == 0:
            img = Image.open(f"{self.dataPath}/negative/{datas[0]}")
        else:
            img = Image.open(f"{self.dataPath}/part/{datas[0]}")


        c, x1, y1, x2, y2 = float(datas[1]), float(datas[2]), float(datas[3]), float(datas[4]), float(datas[5])
        ofSLeye_x, ofSLeye_y = float(datas[6]), float(datas[7])
        ofSReye_x, ofSReye_y = float(datas[8]), float(datas[9])
        ofSNose_x, ofSNose_y = float(datas[10]), float(datas[11])
        ofSLmouth_x, ofSLmouth_y = float(datas[12]), float(datas[13])
        ofSRmouth_x, ofSRmouth_y = float(datas[14]), float(datas[15])

        return tf(img), np.array(
            [c, x1, y1, x2, y2, ofSLeye_x, ofSLeye_y, ofSReye_x, ofSReye_y, ofSNose_x, ofSNose_y, ofSLmouth_x,
             ofSLmouth_y, ofSRmouth_x,
             ofSRmouth_y], dtype=np.float32)


if __name__ == '__main__':
    mydataset = Mydataset("D:\pycharm_workspace\mtcnn_data", 12)
    print(len(mydataset))
