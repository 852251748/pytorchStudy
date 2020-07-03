from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

tf = transforms.Compose([
    transforms.ToTensor()
])


class Mydataset(Dataset):
    def __init__(self, root, img_size):
        self.root = root
        positive_lable = f"{root}/{img_size}/positive.txt"
        negative_lable = f"{root}/{img_size}/negative.txt"
        part_lable = f"{root}/{img_size}/part.txt"
        self.data = []
        with open(positive_lable) as f:
            self.data.extend(f.readlines())
        with open(negative_lable) as f:
            self.data.extend(f.readlines())
        with open(part_lable) as f:
            self.data.extend(f.readlines())
        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        str = self.data[item]
        data = str.split()
        img = Image.open(data[0])
        img = tf(img)

        conf, off_x1, off_y1, off_x2, off_y2 = float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(
            data[5])
        off_lefteye_x, off_lefteye_y, off_righteye_x, off_righteye_y, off_nose_x, off_nose_y, off_leftmouth_x, off_leftmouth_y, off_rightmouth_x, off_rightmouth_y = float(
            data[6]), float(data[7]), float(data[8]), float(data[9]), float(
            data[10]), float(data[11]), float(data[12]), float(data[13]), float(data[14]), float(
            data[15])
        return img, np.array(
            [conf, off_x1, off_y1, off_x2, off_y2, off_lefteye_x, off_lefteye_y, off_righteye_x, off_righteye_y,
             off_nose_x, off_nose_y, off_leftmouth_x, off_leftmouth_y, off_rightmouth_x, off_rightmouth_y],
            dtype=np.float32)


if __name__ == '__main__':
    dataset = Mydataset(r"E:\mtcnn_data", 48)
    print(dataset[0])
    print(dataset[0][0].shape)
