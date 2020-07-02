import numpy as np
import os
import shutil
from PIL import Image
from utils import *
import torch


class GenData:
    def __init__(self, root, imgsize):
        self.imgpath = r"D:\BaiduNetdiskDownload\CelebA\Img\img_celeba.7z\img_celeba"
        self.lableBoxPath = r"D:\BaiduNetdiskDownload\CelebA\Anno\list_bbox_celeba.txt"
        self.lableLandMaPath = r"D:\BaiduNetdiskDownload\CelebA\Anno\list_landmarks_celeba.txt"
        self.imageSize = imgsize

        self.positiveDir = f"{root}/{imgsize}/positive"
        self.negativeDir = f"{root}/{imgsize}/negative"
        self.partDir = f"{root}/{imgsize}/part"
        self.positiveLabel = f"{root}/{imgsize}/positive.txt"
        self.negativeLabel = f"{root}/{imgsize}/negative.txt"
        self.partLabel = f"{root}/{imgsize}/part.txt"

        if not os.path.exists(self.positiveDir):
            os.makedirs(self.positiveDir)

        if not os.path.exists(self.negativeDir):
            os.makedirs(self.negativeDir)

        if not os.path.exists(self.partDir):
            os.makedirs(self.partDir)

    def __call__(self, epoch):
        positiveLabelTxt = open(self.positiveLabel, "w")
        negativeLabelTxt = open(self.negativeLabel, "w")
        partLabelTxt = open(self.partLabel, "w")

        # 读取五个标志点
        landMarks = open(self.lableLandMaPath).readlines()
        landMarkList = []
        for j in range(len(landMarks)):
            landMarkList.append(landMarks[j].split())

        for i, lable in enumerate(open(self.lableBoxPath)):
            if i > 1:
                lableList = lable.split()
                x, y, w, h = int(lableList[1].strip()), \
                             int(lableList[2].strip()), \
                             int(lableList[3].strip()), \
                             int(lableList[4].strip())

                # 原标注框不太标准重新进行标注
                x1, y1, x2, y2 = int(x + 0.12 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.85 * h)
                x, y, w, h = x1, y1, int(x2 - x1), int(y2 - y1)
                cx, cy = (x + w / 2), (y + h / 2)
                # print("32", x1, y1, x2, y2)

                landMark = np.zeros(11)
                if lableList[0] == landMarkList[i][0]:
                    landMark = landMarkList[i]

                img = Image.open(self.imgpath + "/" + lableList[0])
                wide, high = img.size
                poList = []

                # 生成正样本点
                low, high = int(-w * 0.2), int(w * 0.2)
                low1, high1 = int(-h * 0.2), int(h * 0.2)
                if low < high & low1 < high1:
                    sampCx, sampCy = cx + np.random.randint(low, high), \
                                     cy + np.random.randint(low1, high1)

                    sideLen = np.random.randint(int(min(w, h) * 0.9), int(max(w, h) * 1.1))

                    px1, py1 = max(int(sampCx - sideLen / 2), 0), max(int(sampCy - sideLen / 2), 0)
                    px2, py2 = px1 + sideLen, py1 + sideLen

                    poList.append([px1, py1, px2, py2, sideLen])

                # 生成负样本点
                if self.imageSize >= int(min(w, h) / 2):
                    continue
                sideLen = np.random.randint(self.imageSize, int(min(w, h) / 2))
                nx1, ny1 = np.random.randint(0, w - sideLen), np.random.randint(0, h - sideLen)
                nx2, ny2 = nx1 + sideLen, ny1 + sideLen

                poList.append([nx1, ny1, nx2, ny2, sideLen])

                for k in range(len(poList)):
                    # 截取样本图像

                    cropImg = img.crop(poList[k][:4])

                    # 重置样本图片大小
                    cropImg = cropImg.resize((self.imageSize, self.imageSize))
                    # 计算样本标注框的偏移量

                    offSetPx1, offSetPy1, offSetPx2, offSetPy2 = (x1 - poList[k][0]) / poList[k][4], \
                                                                 (y1 - poList[k][1]) / poList[k][4], \
                                                                 (x2 - poList[k][2]) / poList[k][4], \
                                                                 (y2 - poList[k][3]) / poList[k][4]
                    # print("2", poList[k][1], poList[k][0], poList[k][4], poList[k][5])
                    # print("1",offSetPx1, offSetPy1, offSetPx2, offSetPy2)

                    # 计算样本五官的偏移量
                    ofSLeye_x, ofSLeye_y = max((int(landMark[1].strip()) - poList[k][0]) / poList[k][4], 0), \
                                           max((int(landMark[2].strip()) - poList[k][1]) / poList[k][4], 0)
                    ofSReye_x, ofSReye_y = max((int(landMark[3].strip()) - poList[k][0]) / poList[k][4], 0), \
                                           max((int(landMark[4].strip()) - poList[k][1]) / poList[k][4], 0)
                    ofSNose_x, ofSNose_y = max((int(landMark[5].strip()) - poList[k][0]) / poList[k][4], 0), \
                                           max((int(landMark[6].strip()) - poList[k][1]) / poList[k][4], 0)
                    ofSLmouth_x, ofSLmouth_y = max((int(landMark[7].strip()) - poList[k][0]) / poList[k][4], 0), \
                                               max((int(landMark[8].strip()) - poList[k][1]) / poList[k][4], 0)
                    ofSRmouth_x, ofSRmouth_y = max((int(landMark[9].strip()) - poList[k][0]) / poList[k][4], 0), \
                                               max((int(landMark[10].strip()) - poList[k][1]) / poList[k][4], 0)

                    # 计算样本点IOU
                    iou = IOU(torch.tensor([x1, y1, x2, y2]).float(), torch.tensor([poList[k][:4]]).float())

                    # 大于0.65为正样本
                    if iou > 0.65:
                        # 样本图片文件名
                        cropImgName = f"{len(os.listdir(self.positiveDir)) + 1}.jpg"
                        # print(np.array([x1, y1, x2, y2]), np.array(poList[k][:4]), cropImgName, iou)
                        # 保存样本图片
                        cropImg.save(self.positiveDir + "/" + cropImgName)
                        # 保存标签
                        positiveLabelTxt.write(
                            f"{cropImgName} 1 {offSetPx1} {offSetPy1} {offSetPx2} {offSetPy2} {ofSLeye_x} {ofSLeye_y} {ofSReye_x} {ofSReye_y} {ofSNose_x} {ofSNose_y} {ofSLmouth_x} {ofSLmouth_y} {ofSRmouth_x} {ofSRmouth_y}\n")
                        positiveLabelTxt.flush()
                        # print(poList[k])

                    elif iou > 0.4:  # 在0.4和0.65之间为部分样本
                        # 样本图片文件名
                        cropImgName = f"{len(os.listdir(self.partDir)) + 1}.jpg"
                        # print(np.array([x1, y1, x2, y2]), np.array(poList[k][:4]), cropImgName, iou)
                        # 保存样本图片
                        cropImg.save(self.partDir + "/" + cropImgName)
                        # 保存标签
                        partLabelTxt.write(
                            f"{cropImgName} 2 {offSetPx1} {offSetPy1} {offSetPx2} {offSetPy2} {ofSLeye_x} {ofSLeye_y} {ofSReye_x} {ofSReye_y} {ofSNose_x} {ofSNose_y} {ofSLmouth_x} {ofSLmouth_y} {ofSRmouth_x} {ofSRmouth_y}\n")
                        partLabelTxt.flush()
                    elif iou < 0.3:  # 小于0.3为负样本
                        # 样本图片文件名
                        cropImgName = f"{len(os.listdir(self.negativeDir)) + 1}.jpg"
                        # 保存样本图片
                        cropImg.save(self.negativeDir + "/" + cropImgName)
                        # 保存标签
                        negativeLabelTxt.write(
                            f"{cropImgName} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                        negativeLabelTxt.flush()

        # 关闭标签文件
        positiveLabelTxt.close()
        negativeLabelTxt.close()
        partLabelTxt.close()

    def GenerateSample(self):
        pass


if __name__ == '__main__':
    gendata = GenData("E:/mtcnn_data", 48)
    gendata(10)
