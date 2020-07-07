import torch
from deeplearnCode.Day4.Mtcnn.net import PNet, RNet, ONet
from PIL import Image, ImageDraw
from deeplearnCode.Day4.Mtcnn.dataset import tf
from deeplearnCode.Day4.Mtcnn.utils import nms


class Detection:
    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("../param/10_pnet.pt"))
        self.rnet = RNet()
        # self.rnet.load_state_dict(torch.load("../param/65_pnet.pt"))
        self.onet = ONet()
        # self.onet.load_state_dict(torch.load("../param/65_pnet.pt"))

    def __call__(self, img):
        boxes = self.det_Pnet(img)
        if boxes is None: return []

        # boxes = self.det_Rnet(img, boxes)
        # if boxes is None: return []
        #
        # boxes = self.det_Onet(img, boxes)
        # if boxes is None: return []
        return boxes

    def det_Pnet(self, img):
        scale = 1
        scale_img = img
        w, h = img.size
        min_side = min(w, h)

        boxes = []
        while min_side > 12:
            pre = self.pnet(tf(scale_img)[None, ...])
            print(pre.shape)

            # 置信度
            torch.sigmoid_(pre[0, 0])
            mask = pre[0, 0] > 0.65
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

            boxes.append(nms(_boxes, 0.3))

            # 图像金字塔
            scale *= 0.702
            _w, _h = int(w * scale), int(h * scale)
            scale_img = img.resize((_w, _h))
            min_side = min(_w, _h)

        return torch.cat(boxes, dim=0)

    def det_Rnet(self, img, boxes):
        boxes = self.det_R_ONet(img, boxes, 24)
        return nms(boxes, 0.3)

    def det_Onet(self, img, boxes):
        boxes = self.det_R_ONet(img, boxes, 48)
        _boxes = nms(boxes, 0.5)
        return nms(_boxes, 0.5, is_min=True)

    def det_R_ONet(self, img, boxes, s):
        if boxes.shape[0] == 0:
            return torch.tensor([])
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop_img = img.crop((x1, y1, x2, y2))
            resize_img = crop_img.resize((s, s))
            imgs.append(tf(resize_img))

        imgs = torch.stack(imgs, dim=0)
        if s == 24:
            pre = self.rnet(imgs)
        else:
            pre = self.onet(imgs)

        torch.sigmoid_(pre[:, 0])
        torch.tanh_(pre[:, 1:])

        mask = pre[:, 0] > 0.65
        pre_offset = pre[mask]
        _boxes = boxes[mask]

        w, h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        # print(pre_offset.shape, pre_offset[:, 1].shape, _boxes.shape, _boxes[0].shape, w.shape, h.shape)
        x1 = pre_offset[:, 1] * w + _boxes[:, 0]
        y1 = pre_offset[:, 2] * h + _boxes[:, 1]
        x2 = pre_offset[:, 3] * w + _boxes[:, 2]
        y2 = pre_offset[:, 4] * h + _boxes[:, 3]
        conf = pre_offset[:, 0]
        _boxes = torch.stack([x1, y1, x2, y2, conf], dim=1)

        return _boxes


if __name__ == '__main__':
    img = Image.open("0.jpg")
    detec = Detection()
    boxes = detec(img)
    drawImg = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        print("置信度：", box[4])
        drawImg.rectangle((box[0], box[1], box[2], box[3]), outline="red")

    img.show()
    # print(detec(img))
