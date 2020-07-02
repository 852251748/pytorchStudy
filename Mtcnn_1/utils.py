import torch


def IOU(box, boxes, isMin=False):
    boxAre = (box[2] - box[0]) * (box[3] - box[1])
    boxesAre = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w = torch.max(torch.tensor(0), (x2 - x1).long())
    h = torch.max(torch.tensor(0), (y2 - y1).long())
    inter = w * h

    if isMin:
        return inter / torch.min(boxAre, boxesAre).float()
    return inter / (boxesAre + boxAre - inter).float()


def Nms(boxes, threshold, isMin=False):
    if boxes.shape[0] == 0: return []
    _boxes = boxes[(-boxes[:, 0]).argsort()]

    rBoxes = []

    while _boxes.shape[0] > 1:
        box = _boxes[0]
        rBoxes.append(box)

        otherBox = _boxes[1:]

        _boxes = otherBox[(IOU(box[1:5], otherBox[:, 1:5], isMin) < threshold)]

    if _boxes.shape[0] == 1: rBoxes.append(_boxes[0])

    return torch.stack(rBoxes, dim=0)


def ConvertSquare(box):
    square_bbox = box.copy_(box)
    w, h = box[3] - box[1], box[4] - box[2]
    maxSide = max(w, h)  # 112.91  146.98
    cx, cy = box[1] + w / 2, box[2] + h / 2
    square_bbox[1] = cx - maxSide / 2
    square_bbox[2] = cy - maxSide / 2
    square_bbox[3] = cx + maxSide / 2
    square_bbox[4] = cy + maxSide / 2
    return square_bbox


if __name__ == '__main__':
    # a = torch.tensor([3, 3, 5, 5])
    # b = torch.tensor([[1, 1, 3, 3], [2, 2, 4, 4], [3, 3, 5, 5]])
    # # print()
    # print(IOU(a, b) > 0.1)

    # a = torch.tensor([3, 3, 5, 5])
    # b = torch.tensor([[3, 1, 1, 4, 4], [7, 2, 2, 4, 4], [6, 3, 3, 5, 5], [6, 4, 4, 6, 6], [6, 4, 4, 7, 7]])
    # Nms(b, 0.2)
    # box = torch.tensor([1.0000, 190.6342, 213.8967, 303.5801, 360.8767])
    # print(ConvertSquare(box))
    a = torch.tensor([641., 321., 908., 677.])
    b = torch.tensor([[75., 4., 176., 105.]])
    iou = IOU(a, b)
    print(iou)
