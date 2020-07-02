from net import *
import torch
from data import *
from PIL import Image, ImageDraw
from utils import *

DEVICE = "cuda:0"

# 网络调参
# P网络:
p_cls = 0.6  # 原为0.6
p_nms = 0.5  # 原为0.5
# R网络：
r_cls = 0.8  # 原为0.6
r_nms = 0.5  # 原为0.5
# R网络：
o_cls = 0.9  # 原为0.97
o_nms = 0.7  # 原为0.7


class Detect:
    def __init__(self):
        self.pnet = PNet().to(DEVICE)
        self.pnet.load_state_dict(torch.load("./param/pnet.pt"))
        # self.pnet

        self.rnet = RNet().to(DEVICE)
        self.rnet.load_state_dict(torch.load("./param/rnet.pt"))
        # self.rnet
        # #
        self.onet = ONet().to(DEVICE)
        self.onet.load_state_dict(torch.load("./param/onet.pt"))
        # self.onet.

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes.shape[0] == 0: return []
        # return boxes.cpu().detach()

        boxes = self.detRnet(img, boxes)
        if boxes.shape[0] == 0: return []
        return boxes.cpu().detach()

        boxes = self.detOnet(img, boxes)
        if boxes.shape[0] == 0: return []
        return boxes.cpu().detach()

    def detPnet(self, img):
        scale = 1
        scaleimg = img
        w, h = scaleimg.size
        minSide = min(w, h)

        _boxes = []
        while minSide > 12:
            inputimg = tf(scaleimg)
            inputimg = inputimg.to(DEVICE)
            predict = self.pnet(inputimg[None, ...])
            predict = predict.cpu().detach()
            torch.sigmoid_(predict[0, 0])

            mask = predict[0, 0] > p_cls

            index = mask.nonzero()

            x1, y1 = (index[:, 1] * 2) / scale, (index[:, 0] * 2) / scale
            x2, y2 = (index[:, 1] * 2 + 12) / scale, (index[:, 0] * 2 + 12) / scale
            w1, h1 = x2 - x1, y2 - y1

            boxes = predict[0, 1:, mask]
            _x1 = boxes[0, :] * w1 + x1
            _y1 = boxes[1, :] * h1 + y1
            _x2 = boxes[2, :] * w1 + x2
            _y2 = boxes[3, :] * h1 + y2

            cond = predict[0, 0][mask]

            _boxes.append(torch.stack([cond, _x1, _y1, _x2, _y2], dim=1))

            # 图像金字塔
            scale *= 0.7
            _w, _h = int(w * scale), int(h * scale)
            scaleimg = scaleimg.resize((_w, _h))
            minSide = min(_w, _h)
            # break

        boxesT = torch.cat(_boxes, dim=0)
        return Nms(boxesT, p_nms)

    def detRnet(self, img, boxes):
        if boxes.shape[0] == 0: return []
        boxes = self._rnet_onet(img, boxes, 24)
        return Nms(boxes, r_nms)

    def detOnet(self, img, boxes):
        if boxes.shape[0] == 0: return []
        boxes = self._rnet_onet(img, boxes, 48)
        boxes = Nms(boxes, o_nms)
        boxes = Nms(boxes, o_nms, isMin=True)
        return boxes

    def _rnet_onet(self, img, boxes, s):

        imgs = []

        for box in boxes:
            box = ConvertSquare(box)
            cropImg = img.crop(box[1:].cpu().detach().numpy())
            cropImg = cropImg.resize((s, s))
            imgs.append(tf(cropImg))
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(DEVICE)


        if s == 24:
            predict = self.rnet(imgs)
        else:
            predict = self.onet(imgs)
        predict = predict.cpu().detach()
        torch.sigmoid_(predict[:, 0])

        if s == 24:
            mask = predict[:, 0] > r_cls
        else:
            mask = predict[:, 0] > o_cls

        print(mask)
        _boxes = boxes[mask]
        preOf = predict[mask]

        w, h = (_boxes[:, 3] - _boxes[:, 1]), (_boxes[:, 4] - _boxes[:, 2])
        x1 = preOf[:, 1] * w + _boxes[:, 1]
        y1 = preOf[:, 2] * h + _boxes[:, 2]
        x2 = preOf[:, 3] * w + _boxes[:, 3]
        y2 = preOf[:, 4] * h + _boxes[:, 4]
        cond = predict[mask][:, 0]

        if s == 48:
            # 五官偏移量
            ofSLeye_x, ofSLeye_y = preOf[:, 5] * w + _boxes[:, 1], preOf[:, 6] * h + _boxes[:, 2]
            ofSReye_x, ofSReye_y = preOf[:, 7] * w + _boxes[:, 1], preOf[:, 8] * h + _boxes[:, 2]
            ofSNose_x, ofSNose_y = preOf[:, 9] * w + _boxes[:, 1], preOf[:, 10] * h + _boxes[:, 2]
            ofSLmouth_x, ofSLmouth_y = preOf[:, 11] * w + _boxes[:, 1], preOf[:, 12] * h + _boxes[:, 2]
            ofSRmouth_x, ofSRmouth_y = preOf[:, 13] * w + _boxes[:, 1], preOf[:, 14] * h + _boxes[:, 2]
            cond = predict[mask][:, 0]

            return torch.stack(
                [cond, x1, y1, x2, y2, ofSLeye_x, ofSLeye_y, ofSReye_x, ofSReye_y, ofSNose_x, ofSNose_y, ofSLmouth_x,
                 ofSLmouth_y, ofSRmouth_x, ofSRmouth_y], dim=1)

        return torch.stack([cond, x1, y1, x2, y2], dim=1)


if __name__ == '__main__':
    detect = Detect()
    img = Image.open("./0.jpg")
    boxes = detect(img)
    drawImg = ImageDraw.Draw(img)
    # print(boxes)
    for i, box in enumerate(boxes):
        drawImg.rectangle((box[1], box[2], box[3], box[4]), outline="red")

    img.show()
