import numpy as np
import torch


def Iou(box, boxes, isMin=False):
    boxArea = abs((box[0] - box[2]) * (box[1] - box[3]))

    boxesArea = abs((boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3]))

    x1 = np.maximum(box[0], boxes[:, 0])  # 左上角x
    x2 = np.minimum(box[2], boxes[:, 2])  # 右下角x
    y1 = np.maximum(box[1], boxes[:, 1])  # 左上角y
    y2 = np.minimum(box[3], boxes[:, 3])  # 右下角y

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    andArea = w * h

    if isMin:
        return andArea / np.minimum(boxArea, boxesArea)
    return andArea / (boxArea + boxesArea - andArea)


def IouTorch(box, boxes, isMin=False):
    boxArea = (box[0] - box[2]) * (box[1] - box[3])

    boxesArea = abs((boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3]))

    x1 = torch.max(box[0], boxes[:, 0])  # 左上角x
    x2 = torch.min(box[2], boxes[:, 2])  # 右下角x
    y1 = torch.max(box[1], boxes[:, 1])  # 左上角y
    y2 = torch.min(box[3], boxes[:, 3])  # 右下角y

    w = torch.max(torch.tensor(0.), x2 - x1)
    h = torch.max(torch.tensor(0.), y2 - y1)

    andArea = w * h
    # print(boxArea, boxesArea, andArea)
    if isMin:
        return andArea / torch.min(boxArea, boxesArea)
    return andArea / (boxArea + boxesArea - andArea)


def Nms(boxes, threshold, isMin=False):
    if boxes.shape[0] == 0: return []
    sortBox = boxes[(-boxes[:, 0]).argsort()]

    rBox = []
    while sortBox.shape[0] > 1:
        box = sortBox[0]
        rBox.append(box)

        otherBoxes = sortBox[1:]

        sortBox = otherBoxes[Iou(box[1:5].float(), otherBoxes[:, 1:5], isMin) < threshold]

    if sortBox.shape[0] == 1:  rBox.append(sortBox[0])
    return torch.stack(rBox, dim=0)


def IouTorchNew(box, boxes, isMin=False):
    boxArea = (box[0] - box[2]) * (box[1] - box[3])

    boxesArea = (boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3])

    x1 = np.maximum(box[0], boxes[:, 0])  # 左上角x
    x2 = np.minimum(box[2], boxes[:, 2])  # 右下角x
    y1 = np.maximum(box[1], boxes[:, 1])  # 左上角y
    y2 = np.minimum(box[3], boxes[:, 3])  # 右下角y
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    andArea = w * h
    # print(boxArea, boxesArea, andArea)
    if isMin:
        return andArea / np.minimum(boxArea, boxesArea)
    return andArea / (boxArea + boxesArea - andArea)


def NmsNew(boxes, threshold, isMin=False):
    if boxes.shape[0] == 0: return []
    sortBox = boxes[(-boxes[:, 0]).argsort()]

    rBox = []
    while sortBox.shape[0] > 1:
        box = sortBox[0]
        rBox.append(box)

        otherBoxes = sortBox[1:]

        sortBox = otherBoxes[IouTorchNew(box, otherBoxes, isMin) < threshold]

    if sortBox.shape[0] > 0:  rBox.append(sortBox[0])
    return np.stack(rBox)


# 扩充：找到中心点，及最大边长，沿着最大边长的两边扩充
def Convert_to_square(bbox):  # 将长方向框，补齐转成正方形框
    square_bbox = np.copy(bbox)
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]  # 框高
    w = bbox[:, 2] - bbox[:, 0]  # 框宽
    max_side = np.maximum(h, w)  # 返回最大边长
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side  # 加最大边长，加最大边长，决定了正方形框
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox


if __name__ == '__main__':
    box = torch.tensor([1, 1, 3, 3]).float()
    boxes = torch.tensor([[0.7002, 48.2474, 29.2295, 61.3534, 41.2422, 50.4275, 33.9288, 57.7938,
                           33.9820, 53.6926, 36.4402, 50.2968, 38.2597, 56.7710, 38.3974],
                          [0.7023, 36.4839, 31.1258, 49.4534, 43.3316, 38.7758, 35.5726, 47.0387,
                           35.3256, 41.9896, 38.0494, 39.1209, 40.2342, 46.3822, 40.1103],
                          [0.9223, 38.6844, 31.0468, 51.0769, 43.0482, 41.3196, 35.4362, 49.0018,
                           35.3373, 44.8091, 37.9168, 41.5328, 39.9104, 48.4801, 39.9213],
                          [0.9550, 40.5287, 30.9187, 52.7813, 42.7655, 42.7933, 35.1871, 50.0996,
                           35.0942, 45.9701, 37.7608, 43.2491, 39.5631, 49.9739, 39.4463],
                          [0.9426, 42.5697, 30.8312, 54.2740, 42.2339, 44.8671, 34.8822, 52.0094,
                           34.7742, 47.8834, 37.3473, 45.4138, 39.0953, 51.9303, 39.0528],
                          [0.9320, 44.7082, 30.7696, 56.2966, 42.2212, 47.2965, 34.9134, 54.1494,
                           34.5802, 50.1723, 37.2226, 47.9317, 39.0655, 54.1872, 38.8505],
                          [0.9452, 46.5378, 30.6164, 58.2300, 42.1223, 49.0647, 34.7723, 55.9708,
                           34.4308, 52.2120, 37.1329, 49.6025, 38.9044, 55.9556, 38.7611],
                          [0.9637, 48.4288, 30.5618, 60.2306, 42.1495, 50.6876, 34.6936, 57.1575,
                           34.6943, 53.7079, 37.2817, 50.7939, 38.9014, 56.8088, 38.9753],
                          [0.9130, 50.2577, 30.5829, 62.3763, 42.2666, 52.4112, 34.8230, 59.3310,
                           34.9186, 55.0443, 37.4747, 52.2228, 39.0985, 58.4578, 39.3216],
                          [0.8860, 52.3722, 30.6802, 64.1375, 42.0613, 54.8418, 34.8875, 61.6984,
                           34.9849, 57.4983, 37.4460, 54.4788, 39.0387, 60.7839, 39.2327]]).float()
    print(boxes.shape)

    print(NmsNew(boxes, 0.6, True))

    # print(Nms(boxes, 0.1))
    # print(Iou(box, boxes, False))
