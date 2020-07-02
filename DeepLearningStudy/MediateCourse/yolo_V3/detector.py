import torch
from MediateCourse.yolo_V3.tools import nms
from MediateCourse.yolo_V3.net import MainNet
# from MediateCourse.yolo_V3.model import MainNet
from PIL import Image, ImageDraw
from torchvision import transforms
from MediateCourse.yolo_V3 import cfg

tf = transforms.Compose([
    transforms.ToTensor()
])


class Detector(torch.nn.Module):
    def __init__(self, save_path):
        super().__init__()
        # 创建网络
        self.net = MainNet()
        # 加载权重
        self.net.load_state_dict(torch.load(save_path))
        # 设置网络是在测试模式
        self.net.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net.to(device)

    # 侦测过程，入参：图片数据，阀值，anchorsbox框
    def forward(self, input, thresh, anchors):
        # 将图片数据输进网络
        output13, output26, output52 = self.net(input)
        # 过滤掉不同网格中置信度小于阀值的框
        index_13, data_13 = self._filter(output13, 0.5)
        out_13 = self._prase(index_13, data_13, 32, anchors[13])
        index_26, data_26 = self._filter(output26, 0.5)
        out_26 = self._prase(index_26, data_26, 16, anchors[26])
        index_52, data_52 = self._filter(output52, 0.9)
        out_52 = self._prase(index_52, data_52, 8, anchors[52])
        return torch.cat([out_13, out_26, out_52], dim=0)

    def _filter(self, output, thresh):
        # 对网络的输出进行转置，网络输出的格式为(N,C,H,W)，转换成(N,H,W,C)
        output = output.permute(0, 2, 3, 1)
        # 将anchorsbox框的个数和置信度、坐标偏移、高宽比、类别分开
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # 过滤置信度小于阀值的
        mask = output[..., 0] > thresh
        # 获取索引
        index = mask.nonzero()
        # 获取置信度、坐标偏移、高宽比、类别数据
        data = output[mask]
        return index, data

    def _prase(self, index, data, step, anchor):
        # 将anchor转换成tensor
        anchor = torch.Tensor(anchor)
        # 获取置信度
        confidence = data[:, 0]
        # 获取当前尺寸网格下是第几个anchorbox框
        a = index[:, 3]
        # 获取类别
        cls = data[..., 5:]

        if len(cls) == 0:
            cls = torch.Tensor([])
        else:
            cls = torch.argmax(cls, dim=1).float()

        # index与data都是两个维度index.shape=[n,4],data.shape=[n,8]
        # index中是索引位置，第一维度代表第几张图片，第二维度代表的是13*13的网格的第几行，也就是数据中的cy的索引，
        # 第三维度代表是13*13的网格的第几列，也就是数据中的cx的索引，第四维度代表的是第几个anchor框
        # 进行反算cx/cy= (index+offset)*step
        cx = (index[:, 2].float() + data[:, 1]) * step
        cy = (index[:, 1].float() + data[:, 2]) * step
        # 进行反算 w= torch.exp(p_w)*anchorbox框的宽
        #         h= torch.exp(p_h)*anchorbox框的高
        w = anchor[a, 0] * torch.exp(data[:, 3])
        h = anchor[a, 1] * torch.exp(data[:, 4])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = x1 + w
        y2 = y1 + h

        out = torch.stack([confidence, x1, y1, x2, y2, cls], dim=1)

        return out


if __name__ == '__main__':

    save_path = r"./param/520.pkl"
    detector = Detector(save_path)

    img = Image.open(r"./data/path/10.jpg")
    imgdata = tf(img)
    imgdata = imgdata[None, ...]

    out_value = detector(imgdata, 0.3, cfg.ANCHORS_GROUP)
    boxes = []

    # 将相同类别的框做nms最大值抑制
    for j in range(10):
        classify_mask = (out_value[..., -1] == j)
        _boxes = out_value[classify_mask]
        boxes.append(nms(_boxes, 0.2))
    # boxes中的每个元素包含了一个类别中的多个框
    for box in boxes:
        # 画出一个类别中的所有框
        for boxx in box:
            try:
                img_draw = ImageDraw.ImageDraw(img)
                c, x1, y1, x2, y2 = boxx[0:5]
                print(c, x1, y1, x2, y2, boxx[5])
                img_draw.rectangle((x1, y1, x2, y2), outline="red")
                if boxx[5].item() == 0:
                    img_draw.text((x1, y1), "Cat", fill="black")
                else:
                    img_draw.text((x1, y1), "Dog", fill="black")
            except:
                continue
    img.show()
