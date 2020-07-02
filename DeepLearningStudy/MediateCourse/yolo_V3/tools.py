import torch


def iou(box, boxes, isMin=False):
    if boxes.shape[0] == 0:
        return box

    box_area = (box[3] - box[1]) * (box[4] - box[2])
    boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

    x1 = torch.max(box[1], boxes[:, 1])
    y1 = torch.max(box[2], boxes[:, 2])
    x2 = torch.min(box[3], boxes[:, 3])
    y2 = torch.min(box[4], boxes[:, 4])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = (w * h).float()

    if isMin:
        return inter / torch.min(boxes_area, box_area)
    else:
        return inter / (box_area + boxes_area - inter)


def nms(boxes, thresh):
    if boxes.shape[0] == 0:
        return torch.tensor([])

    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_box = []
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        r_box.append(a_box)
        o_box = _boxes[1:]
        _boxes = o_box[iou(a_box, o_box) < thresh]

    if _boxes.shape[0] == 1:
        r_box.append(_boxes[0])

    return torch.stack(r_box, dim=0)


if __name__ == '__main__':
    # box = torch.tensor([0, 1, 1, 3, 6])
    # boxes = torch.tensor([[0, 1, 1, 3, 6], [0, 2, 2, 5, 7], [0, 2, 1, 5, 5]])
    box = torch.tensor([0, 2, 2, 5, 7])
    boxes = torch.tensor([[0, 2, 1, 5, 5]])
    print(iou(box, boxes))
    # print(boxes.shape)
    # print(nms(boxes, 0.3))
