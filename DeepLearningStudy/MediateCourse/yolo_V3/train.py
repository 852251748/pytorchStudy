from MediateCourse.yolo_V3.net import MainNet
# from MediateCourse.yolo_V3.model import MainNet
from MediateCourse.yolo_V3.data import Mydataset
from torch import nn
import torch
from torch.utils.data import DataLoader
import os


def loss_fn(output, target, alpha):
    # 置信度损失函数
    conf_loss_fn = nn.BCEWithLogitsLoss()
    # 偏移量损失函数
    offset_loss_fn = nn.MSELoss()
    # 分类损失函数
    cls_loss_fn = nn.CrossEntropyLoss()
    # cls_loss_fn = nn.BCEWithLogitsLoss()

    # 网络输出的格式为(N,C,H,W)，标签的格式为(N,H,W,C),需要将其中一个的格式转换成相同的，这里选择的是转换网络输出的格式
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    # 需要将网络输出放到CPU上，在反向传播时需要传入double，所以网络输出需要转换成double类型
    output = output.cpu().double()
    # 获取置信度大于0（有对象）的网格数据
    mask_obj = target[..., 0] > 0
    output_obj = output[mask_obj]
    target_obj = target[mask_obj]
    # 有对象的置信度损失
    loss_conf_obj = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    # 有对象的偏移量损失
    loss_offset_obj = offset_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])
    # 有对象的分类损失
    loss_cls_obj = cls_loss_fn(output_obj[:, 5:], torch.argmax(target_obj[:, 5:], dim=1))
    # loss_cls_obj = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5:])
    # 有对象的损失
    loss_obj = loss_conf_obj + loss_offset_obj + loss_cls_obj

    # 获取置信度等于0（无对象）的网格数据
    mask_noobj = target[..., 0] == 0
    output_noobj = output[mask_noobj]
    target_noobj = target[mask_noobj]
    # 无对象的置信度损失，因为没有对象所以不需要学习偏移量和类别
    loss_noobj = conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])

    # alpha控制正负样本比例，正样本较少时增大alpha，增大损失加强对它的惩罚
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss


if __name__ == '__main__':
    # 设置权重保存路径
    save_path = "param"
    # 创建数据集
    dataset = Mydataset(r"./data", "animal_label.txt")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # 创建网络
    net = MainNet()
    # # 判断权重文件是否存在
    # if os.path.exists(save_path):
    #     net.load_state_dict(torch.load(f"{save_path}/640.pkl"))
    # else:
    #     print("No Param")
    # 判断是否存在cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)
    # 创建优化器
    opt = torch.optim.Adam(net.parameters())

    epoch = 0
    while True:
        for target_13, target_26, target_52, img in dataloader:
            img = img.to(device)

            output13, output26, output52 = net(img)

            loss_13 = loss_fn(output13, target_13, 0.5)
            loss_26 = loss_fn(output26, target_26, 0.5)
            loss_52 = loss_fn(output52, target_52, 0.5)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss.item())
        epoch += 1
        if epoch % 20 == 0:
            torch.save(net.state_dict(), f"{save_path}/{epoch}.pkl")
            print('save{}'.format(epoch))
