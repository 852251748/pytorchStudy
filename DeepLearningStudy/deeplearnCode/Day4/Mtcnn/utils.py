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


if __name__ == '__main__':
    box = torch.tensor([1., 2., 8., 9.])
    boxes = torch.tensor([[1., 2., 8., 9.], [4., 5., 8., 9.], [2., 4., 7., 8.]])
    print(Iou(box, boxes,True))
