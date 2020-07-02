from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from MediateCourse.yolo_V3 import cfg
import math

tf = transforms.Compose([
    transforms.ToTensor()
])


def oneHot(cls, i):
    label = np.zeros(int(cls))
    label[i] = 1.
    return label


class Mydataset(Dataset):
    def __init__(self, root, lableFile):
        # 保存数据路径
        self.root = root
        # 保存标签路径
        self.lable_path = f"{self.root}/{lableFile}"
        # 按行读取所有标签
        with open(self.lable_path, "r") as f:
            self.lable = f.readlines()

    def __len__(self):
        return len(self.lable)

    def __getitem__(self, item):
        data = self.lable[item]
        # 将标签按照空格分隔
        strs = data.split()
        # 打开图片
        img = Image.open(f"{self.root}/{strs[0]}")
        # 转换成Tensor 并且进行归一化
        img = tf(img)
        # 使用列表推导式将坐标和分类取出
        _boxes = np.array([float(x) for x in strs[1:]])
        # 按类别、中心点x、中心点y、宽、高 五个数据进行分组
        boxes = np.split(_boxes, len(_boxes) // 5)

        labels = {}
        # 遍历自定义的anchorbox
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            # 创建空的特征图，将对象的中心点数据填入对应的网格内
            labels[feature_size] = np.zeros((feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            # 遍历实际的框
            for box in boxes:
                # 获取实际的类别、中心点x、中心点y、宽、高
                cls, cx, cy, w, h = box
                # 计算对象中心点（cx,cy）在网格中的索引和相对于对应网格左上角的偏移量 少用除法 容易丢失数据
                # cx_offet, cx_index = math.modf(cx / cfg.IMG_WIDTH / feature_size)
                # cy_offet, cy_index = math.modf(cy / cfg.IMG_HEIGHT / feature_size)
                cx_offet, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offet, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)
                # 遍历每个尺寸中anchorbox的框
                for i, anchor in enumerate(anchors):
                    # 计算anchorbox框的面积
                    anchor_area = anchor[0] * anchor[1]  # 直接使用anchorbox的高宽相乘
                    # 计算实际框的面积
                    act_area = w * h
                    # 实际框和anchorbox框面积的最小值/实际框和anchorbox框面积的最小值 的值做为置信度
                    conf = min(anchor_area, act_area) / max(anchor_area, act_area)
                    # 使用实际框的宽(高)/anchorbox框的宽(高)的值加个log学习 值在1附近 log函数的梯度较大容易学习，值域是负无穷到正无穷符合网络输出，不用加激活函数
                    p_w, p_h = np.log(w / anchor[0]), np.log(h / anchor[1])
                    # 在中心点对应的网格中填入置信度，偏移量，宽，高比值，以及类别的onehot格式
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [conf, cx_offet, cy_offet, p_w, p_h, *oneHot(cfg.CLASS_NUM, int(cls))])  # oneHot函数前加'*'是解压返回的列表

        return labels[13], labels[26], labels[52], img


if __name__ == '__main__':
    dataset = Mydataset(r"./data", "person_lable.txt")
    label13, label26, label52, img = dataset[0]
    print(label13.shape, label26.shape, label52.shape, img.shape)
