import cv2 as cv
import os
from PIL import ImageDraw, Image

datapath = r"E:/mtcnn_data/24/positive.txt"
fp = open(datapath)
lableBoxPath = r"D:\BaiduNetdiskDownload\CelebA\Anno\list_bbox_celeba.txt"
actpicture = r"D:\BaiduNetdiskDownload\CelebA\Img\img_celeba.7z\img_celeba/"

act = open(lableBoxPath).readlines()
for i, data in enumerate(fp):
    lableList = data.split()

    ofx1, ofy1, ofx2, ofy2 = float(lableList[1].strip()), \
                             float(lableList[2].strip()), \
                             float(lableList[3].strip()), \
                             float(lableList[4].strip())
    x1, y1, w1, h1 = int(act[i + 2].split()[1]), \
                     int(act[i + 2].split()[2]), \
                     int(act[i + 2].split()[3]), \
                     int(act[i + 2].split()[4])

    x11, y11, w11, h11 = int(x1 + 0.12 * w1), int(y1 + 0.1 * h1), int(x1 + 0.9 * w1), int(y1 + 0.85 * h1)

    px1, py1, px2, py2 = ofx1 * 12 + x11, ofy1 * 12 + y11, ofx2 * 12 + w11, ofy2 * 12 + h11

    actfileName = act[i + 2].split()[0]

    img = Image.open(actpicture + actfileName)
    draw = ImageDraw.Draw(img)
    draw.rectangle(( px1, py1, px2, py2), outline='red')
    draw.rectangle((x11, y11, w11, h11), outline='blue')
    img.show()
