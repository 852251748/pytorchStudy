import torch
import numpy as np
from deeplearnCode.Day4.Mtcnn.utils import Iou
from PIL import Image
import os


class Gendata:
    def __init__(self, root, data_path, img_size):
        self.img_size = img_size
        self.img_path = f"{root}/Img/img_celeba"
        self.boxlable_path = f"{root}/Anno/list_bbox_celeba.txt"
        self.landmasklable_path = f"{root}/Anno/list_landmarks_celeba.txt"
        self.positive = f"{data_path}/{img_size}/positive"
        self.negative = f"{data_path}/{img_size}/negative"
        self.part = f"{data_path}/{img_size}/part"
        self.positive_lable = f"{data_path}/{img_size}/positive.txt"
        self.negative_lable = f"{data_path}/{img_size}/negative.txt"
        self.part_lable = f"{data_path}/{img_size}/part.txt"

        if not os.path.exists(self.positive):
            os.mkdir(self.positive)
        if not os.path.exists(self.negative):
            os.mkdir(self.negative)
        if not os.path.exists(self.part):
            os.mkdir(self.part)

    def __call__(self, ):
        # 读取五官的坐标点
        with open(self.landmasklable_path) as lmf:
            landmarsk = lmf.readlines()[2:]
            landmarsk_data = {}
            for str in landmarsk:
                list = str.split()
                landmarsk_data[list[0]] = list[1:]

        with open(self.boxlable_path) as f:
            data = f.readlines()
            for i, str in enumerate(data[100000:]):
                data = str.split()

                x, y, w, h = data[1:5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x1, y1, x2, y2 = int(x + 0.12 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.85 * h)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = int(x1 + w / 2), int(y1 + h / 2)

                sampleList = []
                for i in range(2):
                    # 生正成样本点
                    if int(-w * 0.2) >= int(w * 0.2) or int(-h * 0.2) >= int(h * 0.2):
                        continue
                    sampleCx, sampleCy = cx + np.random.randint(int(-w * 0.2), int(w * 0.2)), cy + np.random.randint(
                        int(-h * 0.2), int(h * 0.2))

                    p_sideLen = np.random.randint(int(min(w, h) * 0.95), int(max(w, h) * 1.2))
                    p_sx1, p_sy1 = max(int(sampleCx - p_sideLen / 2), 0), max(int(sampleCy - p_sideLen / 2), 0)
                    p_sx2, p_sy2 = p_sx1 + p_sideLen, p_sy1 + p_sideLen

                    sampleList.append([p_sx1, p_sy1, p_sx2, p_sy2, p_sideLen])

                    # # 生成负样本
                    # if self.img_size >= int(min(w, h) / 2):
                    #     continue
                    # n_sideLen = np.random.randint(int(self.img_size), int(min(w, h) / 2))
                    # n_sx1, n_sy1 = np.random.randint(0, w - n_sideLen), np.random.randint(0, h - n_sideLen)
                    # n_sx2, n_sy2 = n_sx1 + n_sideLen, n_sy1 + n_sideLen
                    #
                    # sampleList.append([n_sx1, n_sy1, n_sx2, n_sy2, n_sideLen])

                # 计算偏移量
                # 网络要预测是物体的真实框的坐标，偏移量计算的是物体真实框的坐标和裁剪的样本的坐标的偏移值，所以偏移量中除数w/h就是裁剪的图片的宽高
                for sx1, sy1, sx2, sy2, sideLen in sampleList:
                    off_x1, off_y1, off_x2, off_y2 = (x1 - sx1) / sideLen, (y1 - sy1) / sideLen, (x2 - sx2) / sideLen, (
                            y2 - sy2) / sideLen

                    # 计算五官偏移量
                    landmarsk_list = landmarsk_data[data[0]]
                    lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = int(
                        landmarsk_list[0]), int(landmarsk_list[1]), int(landmarsk_list[2]), int(landmarsk_list[3]), int(
                        landmarsk_list[4]), int(landmarsk_list[5]), int(landmarsk_list[6]), int(landmarsk_list[7]), int(
                        landmarsk_list[8]), int(landmarsk_list[9])

                    off_lefteye_x = (lefteye_x - sx1) / sideLen
                    off_lefteye_y = (lefteye_y - sy1) / sideLen
                    off_righteye_x = (righteye_x - sx1) / sideLen
                    off_righteye_y = (righteye_y - sy1) / sideLen
                    off_nose_x = (nose_x - sx1) / sideLen
                    off_nose_y = (nose_y - sy1) / sideLen
                    off_leftmouth_x = (leftmouth_x - sx1) / sideLen
                    off_leftmouth_y = (leftmouth_y - sy1) / sideLen
                    off_rightmouth_x = (rightmouth_x - sx1) / sideLen
                    off_rightmouth_y = (rightmouth_y - sy1) / sideLen

                    # 裁剪图片
                    img = Image.open(f"{self.img_path}/{data[0]}")
                    crop_img = img.crop([sx1, sy1, sx2, sy2])
                    # 重置大小
                    resize_img = crop_img.resize((self.img_size, self.img_size))
                    # 计算IOU
                    iou = Iou(torch.tensor([x1, y1, x2, y2]).float(), torch.tensor([[sx1, sy1, sx2, sy2]]).float())

                    if iou > 0.6:
                        sampleimg_path = f"{self.positive}/{len(os.listdir(self.positive))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.positive_lable, "a") as f1:
                            f1.write(
                                f"{sampleimg_path} 1 {off_x1} {off_y1} {off_x2} {off_y2} {off_lefteye_x} {off_lefteye_y} {off_righteye_x} {off_righteye_y} {off_nose_x} {off_nose_y} {off_leftmouth_x} {off_leftmouth_y} {off_rightmouth_x} {off_rightmouth_y}\n")
                    elif iou > 0.4:
                        sampleimg_path = f"{self.part}/{len(os.listdir(self.part))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.part_lable, "a") as f1:
                            f1.write(
                                f"{sampleimg_path} 2 {off_x1} {off_y1} {off_x2} {off_y2} {off_lefteye_x} {off_lefteye_y} {off_righteye_x} {off_righteye_y} {off_nose_x} {off_nose_y} {off_leftmouth_x} {off_leftmouth_y} {off_rightmouth_x} {off_rightmouth_y}\n")
                    elif iou < 0.3:
                        sampleimg_path = f"{self.negative}/{len(os.listdir(self.negative))}.jpg"
                        resize_img.save(sampleimg_path)
                        with open(self.negative_lable, "a") as f1:
                            f1.write(f"{sampleimg_path} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")


if __name__ == '__main__':
    gendta = Gendata("D:\Alldata\CelebA", "E:\mtcnn_data", 12)
    gendta()
