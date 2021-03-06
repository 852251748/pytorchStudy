import torch


def Iou(box, boxes, is_Min=False):
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w = torch.max((x2 - x1), torch.tensor(0.))
    h = torch.max((y2 - y1), torch.tensor(0.))

    inter_area = w * h

    if is_Min:
        return inter_area / torch.min(area_box, area_boxes)
    return inter_area / (area_box + area_boxes - inter_area)


def nms(boxes, threshold, is_min=False):
    if boxes.shape[0] == 0: return torch.tensor([])
    # 对boxes根据置信度大小进排序 从大到小
    _boxes = boxes[(-boxes[:, -1]).argsort()]
    rboxes = []
    while _boxes.shape[0] > 1:
        first_box = _boxes[0]
        rboxes.append(first_box)
        other_boxes = _boxes[1:]

        index = (Iou(first_box, other_boxes, is_min) < threshold)
        _boxes = other_boxes[index]

    if _boxes.shape[0] > 0:
        rboxes.append(_boxes[0])

    return torch.stack(rboxes, dim=0)


def make_square(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w, h = x2 - x1, y2 - y1
    sidelen = max(w, h)
    cx, cy = (x1 + w / 2), (y1 + h / 2)
    x1, y1 = (cx - sidelen / 2), (cy - sidelen / 2)
    x2, y2 = x1 + sidelen, y1 + sidelen
    return torch.tensor([x1, y1, x2, y2])


if __name__ == '__main__':
    box = torch.tensor([4., 5., 8., 9., 0.7])
    boxes = torch.tensor([[1., 2., 8., 9., 0.5], [4., 5., 8., 9., 0.7], [3., 6., 10., 12., 0.6]])
    print(Iou(box, boxes, False))
    print(nms(boxes, 0.35))
