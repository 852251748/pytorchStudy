import torch
from net import PNet, RNet, ONet, PNet2
from PIL import Image, ImageDraw
from dataset import tf
from utils import nms, make_square
from torchvision import transforms
import os

tf1 = transforms.Compose([
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络调参
# P网络:
p_cls = 0.49  # 原为0.6
p_nms = 0.5  # 原为0.5
# R网络：
r_cls = 0.6  # 原为0.6
r_nms = 0.5  # 原为0.5
# O网络：
o_cls = 0.999  # 原为0.97
o_nms = 0.5  # 原为0.7
o_min_nms = 0.5  # 原为0.7


class Detection:
    def __init__(self):

        # self.pnet = PNet()
        # self.pnet.load_state_dict(torch.load("../param/test_param/net.pt"))
        # self.pnet.load_state_dict(torch.load("../param/15_pnet.pt"))
        self.pnet = PNet()
        self.pnet.load_state_dict(
            torch.load(r"F:\PycharmWorkspace\pytorchStudy\DeepLearningStudy\deeplearnCode\Day4\param\60_pnet.pt"))
        # self.pnet.load_state_dict(torch.load("../param/test_param/net.pt"))

        self.pnet.to(device)

        self.rnet = RNet()
        self.rnet.load_state_dict(torch.load("../param/100_rnet.pt"))
        self.rnet.to(device)
        #
        self.onet = ONet()
        self.onet.load_state_dict(torch.load("../param/210_onet.pt"))
        self.onet.to(device)

    def __call__(self, img):
        boxes = self.det_Pnet(img)
        if boxes is None: return []

        boxes = self.det_Rnet(img, boxes)
        if boxes is None: return []

        boxes = self.det_Onet(img, boxes)
        if boxes is None: return []
        return boxes

    def det_Pnet(self, img):
        scale = 1
        scale_img = img
        w, h = img.size
        min_side = min(w, h)

        boxes = []
        while min_side > 12:
            input = tf1(scale_img)[None, ...].to(device)
            print(input.shape)
            pre = self.pnet(input)
            pre = pre.cpu().detach()

            # 置信度
            print(torch.sigmoid_(pre[0, 0]))
            mask = pre[0, 0] > p_cls
            # 偏移量
            offsets = torch.tanh_(pre[0, 1:5, mask])
            # 索引
            indexs = mask.nonzero()

            # 算出建议框的左上角和右下角坐标
            anchor_x1, anchor_y1 = indexs[:, 1] * 2, indexs[:, 0] * 2
            anchor_x2, anchor_y2 = anchor_x1 + 12, anchor_y1 + 12
            # 实际框=偏移量*w+建议框对应的坐标 再出以缩放比例就等于原图上的坐标点
            act_x1 = (offsets[0] * 12 + anchor_x1) / scale
            act_y1 = (offsets[1] * 12 + anchor_y1) / scale
            act_x2 = (offsets[2] * 12 + anchor_x2) / scale
            act_y2 = (offsets[3] * 12 + anchor_y2) / scale
            conf = pre[0, 0, mask]
            # 将坐标点与置信度组合成张量进行Nms
            _boxes = torch.stack([act_x1, act_y1, act_x2, act_y2, conf], dim=1)

            boxes.append(_boxes)

            # 图像金字塔
            scale *= 0.702
            _w, _h = int(w * scale), int(h * scale)
            scale_img = img.resize((_w, _h))
            min_side = min(_w, _h)

        boxes = torch.cat(boxes, dim=0)

        return nms(boxes, p_nms)

    def det_Rnet(self, img, boxes):
        boxes = self.det_R_ONet(img, boxes, 24)
        return nms(boxes, r_nms)

    def det_Onet(self, img, boxes):
        boxes = self.det_R_ONet(img, boxes, 48)
        _boxes = nms(boxes, o_nms)
        return nms(_boxes, o_min_nms, is_min=True)

    def det_R_ONet(self, img, boxes, s):
        if boxes.shape[0] == 0:
            print("1")
            return torch.tensor([])
        imgs = []

        for box in boxes:
            # box = make_square(box)
            # x1, y1, x2, y2 = torch.ceil(box[0]).numpy(), torch.ceil(box[1]).numpy(), torch.ceil(box[2]).numpy(), torch.ceil(box[3]).numpy()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop_img = img.crop((x1, y1, x2, y2))
            resize_img = crop_img.resize((s, s))
            imgs.append(tf(resize_img))

        imgs = torch.stack(imgs, dim=0).to(device)

        if s == 24:
            pre = self.rnet(imgs)
        else:
            pre = self.onet(imgs)

        pre = pre.cpu().detach()
        torch.sigmoid_(pre[:, 0])
        torch.tanh_(pre[:, 1:])
        if s == 24:
            mask = pre[:, 0] > r_cls
        else:
            mask = pre[:, 0] > o_cls

        pre_offset = pre[mask]
        _boxes = boxes[mask]

        w, h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        # print(pre_offset.shape, pre_offset[:, 1].shape, _boxes.shape, _boxes[0].shape, w.shape, h.shape)
        x1 = pre_offset[:, 1] * w + _boxes[:, 0]
        y1 = pre_offset[:, 2] * h + _boxes[:, 1]
        x2 = pre_offset[:, 3] * w + _boxes[:, 2]
        y2 = pre_offset[:, 4] * h + _boxes[:, 3]

        off_lefteye_x = pre_offset[:, 5] * w + _boxes[:, 0]
        off_lefteye_y = pre_offset[:, 6] * h + _boxes[:, 1]
        off_righteye_x = pre_offset[:, 7] * w + _boxes[:, 0]
        off_righteye_y = pre_offset[:, 8] * h + _boxes[:, 1]
        off_nose_x = pre_offset[:, 9] * w + _boxes[:, 0]
        off_nose_y = pre_offset[:, 10] * h + _boxes[:, 1]
        off_leftmouth_x = pre_offset[:, 11] * w + _boxes[:, 0]
        off_leftmouth_y = pre_offset[:, 12] * h + _boxes[:, 1]
        off_rightmouth_x = pre_offset[:, 13] * w + _boxes[:, 0]
        off_rightmouth_y = pre_offset[:, 14] * h + _boxes[:, 1]
        # 预测出的框偏窄
        if s == 24:
            _w, _h = (x2 - x1), (y2 - y1)
            x1, x2 = (x1 - _w * 0.25), (x2 + _w * 0.25)
        conf = pre_offset[:, 0]
        _boxes = torch.stack(
            [x1, y1, x2, y2, conf, off_lefteye_x, off_lefteye_y, off_righteye_x, off_righteye_y, off_nose_x, off_nose_y,
             off_leftmouth_x, off_leftmouth_y, off_rightmouth_x, off_rightmouth_y], dim=1)

        return _boxes


if __name__ == '__main__':
    # for filename in os.listdir("./test_img"):
    #     img = Image.open(f"./test_img/{filename}")
    #     detec = Detection()
    #     boxes = detec(img)
    #     drawImg = ImageDraw.Draw(img)
    #     for i, box in enumerate(boxes):
    #         print("置信度：", box[4])
    #         drawImg.rectangle((box[0], box[1], box[2], box[3]), outline="red")
    #
    #     img.show()
    # print(detec(img))

    img = Image.open(f"./test_img/0.jpg")
    detec = Detection()
    boxes = detec(img)
    drawImg = ImageDraw.Draw(img)
    k = 2
    for i, box in enumerate(boxes):
        print("置信度：", box[4])
        drawImg.rectangle((box[0], box[1], box[2], box[3]), outline="red")
        drawImg.ellipse((int(box[5]) - k, int(box[6]) - k, int(box[5]) + k, int(box[6]) + k), fill='red')
        drawImg.ellipse((int(box[7]) - k, int(box[8]) - k, int(box[7]) + k, int(box[8]) + k), fill='red')
        drawImg.ellipse((int(box[9]) - k, int(box[10]) - k, int(box[9]) + k, int(box[10]) + k), fill='red')
        drawImg.ellipse((int(box[11]) - k, int(box[12]) - k, int(box[11]) + k, int(box[12]) + k), fill='red')
        drawImg.ellipse((int(box[13]) - k, int(box[14]) - k, int(box[13]) + k, int(box[14]) + k), fill='red')
    img.show()
