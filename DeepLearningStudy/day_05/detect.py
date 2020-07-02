# from day_05.net import *
from day_05.net_new import *
import torch
from PIL import Image, ImageDraw
from day_05.data import *
# from day_05.utils import *
from day_05.utils_new import *

DEVICE = "cuda:0"

# P网络参数
pCls = 0.75
pNms = 0.3
# R网络参数
rCls = 0.6
rNms = 0.5
# O网络参数
oCls = 0.97
oNms = 0.7


class Detected:
    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("./pnet.pt"))
        self.pnet.eval()

        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load("./rnet.pt"))
        self.rnet.eval()
        #
        self.onet = ONet()
        self.onet.load_state_dict(torch.load("./onet.pt"))
        self.onet.eval()

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes.shape[0] == 0: return []
        # return boxes

        boxes = self.detRnet(img, boxes)
        if boxes.shape[0] == 0: return []
        # return boxes

        boxes = self.detOnet(img, boxes)
        if boxes.shape[0] == 0: return []
        return boxes

    def detPnet(self, img):
        w, h = img.size
        scaleImg = img
        scale = 1
        minSide = min(w, h)

        _boxex = []
        while minSide > 12:
            tImg = tf(scaleImg)
            # cond1, boxof1, ladmof1 = self.pnet(tImg[None, ...])
            cond1, boxof1 = self.pnet(tImg[None, ...])

            # cond1, boxof1, ladmof1 = cond1.cpu().detach(), boxof1.cpu().detach(), ladmof1.cpu().detach()

            mask = cond1[0, 0] > pCls
            index = mask.nonzero()
            # 建议框在缩放后的图片上的坐标
            _x1, _y1 = (index[:, 1] * 2), (index[:, 0] * 2)
            _x2, _y2 = _x1 + 12, _y1 + 12

            # 有人脸的框的置信度
            con = cond1[0, 0, mask]

            # 预测出有人脸的标注框在原图（未缩放）上的坐标
            boxOf1 = boxof1[0, :, mask]
            x1 = (boxOf1[0] * 12 + _x1) / scale
            y1 = (boxOf1[1] * 12 + _y1) / scale
            x2 = (boxOf1[2] * 12 + _x2) / scale
            y2 = (boxOf1[3] * 12 + _y2) / scale

            # # 预测出有人脸的五官点在原图（未缩放）上的坐标
            # ladmOf1 = ladmof1[0, :, mask]
            # lEyeX = (ladmOf1[0] * 12 + _x1) / scale
            # lEyeY = (ladmOf1[1] * 12 + _y1) / scale
            # rEyeX = (ladmOf1[2] * 12 + _x1) / scale
            # rEyeY = (ladmOf1[3] * 12 + _y1) / scale
            # noseX = (ladmOf1[4] * 12 + _x1) / scale
            # noseY = (ladmOf1[5] * 12 + _y1) / scale
            # lMoX = (ladmOf1[6] * 12 + _x1) / scale
            # lMoY = (ladmOf1[7] * 12 + _y1) / scale
            # rMoX = (ladmOf1[8] * 12 + _x1) / scale
            # rMoY = (ladmOf1[9] * 12 + _y1) / scale

            # _boxex.append(torch.stack(
            #     [con, x1, y1, x2, y2, lEyeX, lEyeY, rEyeX, rEyeY, noseX, noseY, lMoX, lMoY, rMoX, rMoY], dim=1))
            # _boxex.append(torch.stack(
            #     [con, x1, y1, x2, y2], dim=1))
            _boxex.append(torch.stack(
                [x1, y1, x2, y2, con], dim=1))

            # 图像金字塔
            scale *= 0.7
            _w, _h = int(w * scale), int(h * scale)
            scaleImg = scaleImg.resize((_w, _h))
            minSide = min(_w, _h)

        boxes = torch.cat(_boxex, dim=0)
        # return boxes
        return nms(boxes.detach().numpy(), pNms)

    def detRnet(self, img, boxes):
        _boxes = self.detRnetAOnet(img, boxes, 24)
        return nms(_boxes, rNms)

    def detOnet(self, img, boxes):
        boxes = self.detRnetAOnet(img, boxes, 48)
        return nms(boxes, oNms, True)

    def detRnetAOnet(self, img, boxes, imgsize):
        imgs = []
        _pnet_boxes = convert_to_square(boxes)
        for box in _pnet_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropImg = img.crop((x1, y1, x2, y2))
            cropImg = cropImg.resize((imgsize, imgsize))
            cropImg = tf(cropImg)
            imgs.append(cropImg)
        imgs = torch.stack(imgs, dim=0)

        # if imgsize == 24:
        #     cond1, boxof1, ladmof1 = self.rnet(imgs)
        # else:
        #     cond1, boxof1, ladmof1 = self.onet(imgs)
        if imgsize == 24:
            cond1, boxof1 = self.rnet(imgs)
        else:
            cond1, boxof1 = self.onet(imgs)

        if imgsize == 24:
            mask = cond1[:, 0] > rCls
        else:
            mask = cond1[:, 0] > oCls

        index = mask.nonzero()[:, 0]
        cc = cond1[mask][:, 0]
        _boxes = boxes[index]
        _boxOf = boxof1[index]
        w, h = _boxes[:, 3] - _boxes[:, 1], _boxes[:, 4] - _boxes[:, 2]

        # 标注框坐标点
        x1 = _boxOf[:, 0] * w + _boxes[:, 1]
        y1 = _boxOf[:, 1] * h + _boxes[:, 2]
        x2 = _boxOf[:, 2] * w + _boxes[:, 3]
        y2 = _boxOf[:, 3] * h + _boxes[:, 4]
        #
        # _landmOf = ladmof1[index]
        # # 五官坐标点反算
        # lEyeX = _landmOf[:, 0] * w + _boxes[:, 1]
        # lEyeY = _landmOf[:, 1] * h + _boxes[:, 2]
        # rEyeX = _landmOf[:, 2] * w + _boxes[:, 1]
        # rEyeY = _landmOf[:, 3] * h + _boxes[:, 2]
        # noseX = _landmOf[:, 4] * w + _boxes[:, 1]
        # noseY = _landmOf[:, 5] * h + _boxes[:, 2]
        # lMoX = _landmOf[:, 6] * w + _boxes[:, 1]
        # lMoY = _landmOf[:, 7] * h + _boxes[:, 2]
        # rMoX = _landmOf[:, 8] * w + _boxes[:, 1]
        # rMoY = _landmOf[:, 9] * h + _boxes[:, 2]
        # return torch.stack([cc, x1, y1, x2, y2, lEyeX, lEyeY, rEyeX, rEyeY, noseX, noseY, lMoX, lMoY, rMoX, rMoY],
        #                    dim=1)
        return np.stack([x1, y1, x2, y2, cc],
                        axis=1)


if __name__ == '__main__':
    img = Image.open("./4.jpg")
    detected = Detected()
    boxes = detected(img)
    # print(boxes.shape)
    drawImg = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        print("置信度：", box[0])
        drawImg.rectangle((box[1], box[2], box[3], box[4]), outline="red")

    img.show()
