import torch
import numpy as np
from deeplearnCode.Day4.Mtcnn.utils import Iou
from PIL import Image
import os


class Gendata:
    def __init__(self, root, data_path, img_size):
        self.img_size = img_size
        self.img_path = f"{root}/Img/img_celeba"
        self.lable_path = f"{root}/Anno/list_bbox_celeba.txt"
        self.positive = f"{data_path}/{img_size}/positive"
        self.negative = f"{data_path}/{img_size}/negative"
        self.part = f"{data_path}/{img_size}/part"
        self.positive_lable = f"{data_path}/{img_size}/positive.txt"
        self.negative_lable = f"{data_path}/{img_size}/negative.txt"
        self.part_lable = f"{data_path}/{img_size}/part.txt"

    def __call__(self, ):

        with open(self.lable_path) as f:
            data = f.readlines()
            for str in data[2:]:
                data = str.split()
                x, y, w, h = data[1:5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x1, y1, x2, y2 = int(x + 0.12 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.85 * h)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = int(x1 + w / 2), int(y1 + h / 2)

                sampleList = []
                for i in range(2):
                    # 生正成样本点
                    sampleCx, sampleCy = cx + np.random.randint(int(-w * 0.2), int(w * 0.2)), cy + np.random.randint(
                        int(-h * 0.2), int(h * 0.2))

                    p_sideLen = np.random.randint(int(min(w, h) * 0.95), int(max(w, h) * 1.2))
                    p_sx1, p_sy1 = max(int(sampleCx - p_sideLen / 2), 0), max(int(sampleCy - p_sideLen / 2), 0)
                    p_sx2, p_sy2 = p_sx1 + p_sideLen, p_sy1 + p_sideLen

                    sampleList.append([p_sx1, p_sy1, p_sx2, p_sy2, p_sideLen])

                    # 生成负样本
                    if self.img_size >= int(min(w, h) / 2):
                        continue
                    n_sideLen = np.random.randint(int(self.img_size), int(min(w, h) / 2))
                    n_sx1, n_sy1 = np.random.randint(0, w - n_sideLen), np.random.randint(0, h - n_sideLen)
                    n_sx2, n_sy2 = n_sx1 + n_sideLen, n_sy1 + n_sideLen

                    sampleList.append([n_sx1, n_sy1, n_sx2, n_sy2, n_sideLen])

                # 计算偏移量
                # 网络要预测是物体的真实框的坐标，偏移量计算的是物体真实框的坐标和裁剪的样本的坐标的偏移值，所以偏移量中除数w/h就是裁剪的图片的宽高
                for sx1, sy1, sx2, sy2, sideLen in sampleList:
                    off_x1, off_y1, off_x2, off_y2 = (x1 - sx1) / sideLen, (y1 - sy1) / sideLen, (x2 - sx2) / sideLen, (
                            y2 - sy2) / sideLen
                    # 裁剪图片
                    img = Image.open(f"{self.img_path}/{data[0]}")
                    crop_img = img.crop([sx1, sy1, sx2, sy2])
                    # 重置大小
                    resize_img = crop_img.resize((self.img_size, self.img_size))
                    # 计算IOU
                    iou = Iou(torch.tensor([x1, y1, x2, y2]).float(), torch.tensor([[sx1, sy1, sx2, sy2]]).float())

                    if iou > 0.65:
                        sampleimg_path = f"{self.positive}/{len(os.listdir(self.positive))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.positive_lable, "a") as f1:
                            f1.write(f"{sampleimg_path} 1 {off_x1} {off_y1} {off_x2} {off_y2}\n")
                    elif iou > 0.4:
                        sampleimg_path = f"{self.part}/{len(os.listdir(self.part))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.part_lable, "a") as f1:
                            f1.write(f"{sampleimg_path} 0 {off_x1} {off_y1} {off_x2} {off_y2}\n")
                    elif iou < 0.3:
                        sampleimg_path = f"{self.negative}/{len(os.listdir(self.negative))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.negative_lable, "a") as f1:
                            f1.write(f"{sampleimg_path} 0 {off_x1} {off_y1} {off_x2} {off_y2}\n")


if __name__ == '__main__':
    gendta = Gendata("D:\Alldata\CelebA", "E:\mtcnn_data", 48)
    gendta()
