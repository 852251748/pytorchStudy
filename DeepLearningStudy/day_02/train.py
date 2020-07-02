# 训练模型
from torch import optim, nn
from day_02.net import *
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets, transforms

DEVICE = "cuda:0"

transform = transforms.Compose([
    transforms.Pad(4),  # Zero boundary fill for pictures
    transforms.RandomHorizontalFlip(),  # Random horizontal flip with probability of 0.5
    transforms.RandomCrop(32),  # Random clipping of pictures for a given size.
    transforms.ToTensor()
])

config = [[-1, 32, 1, 2],
          [1, 16, 1, 1],
          [6, 24, 2, 2],
          [6, 32, 3, 2],
          [6, 64, 4, 2],
          [6, 96, 3, 1],
          [6, 160, 3, 2],
          [6, 320, 1, 1]]


class Train:
    def __init__(self, root):
        # 创建tensorboard数据记录目录
        self.summaryWriter = SummaryWriter("./logs")
        # 加载训练数据集
        self.train_dataset = datasets.CIFAR10(root, True, transform=transform, download=True)
        # 将60000份数据分成每批次100个数据
        self.train_dataLoader = dataloader.DataLoader(self.train_dataset, batch_size=10, shuffle=True)

        # 加载测试数据集
        self.test_dataset = datasets.CIFAR10(root, False, transform=transforms.ToTensor(), download=True)
        # 将10000份数据分成每批次100个数据
        self.test_dataLoader = dataloader.DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建模型
        self.net = MobileNet1(config)
        # 加载保存的参数
        # self.net.load_state_dict(torch.load("./checkpoint/1.pkl"))
        self.net.to(DEVICE)

        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

        # CrossEntropyLoss(pred_y,tags)步骤：
        # 1. y1 = 对pred_y做softmax处理（值都在0-1之间）
        # 2. y2 = 对y1进行取对数（加个log）（值都为负的），
        # 3.再将y2和tags传入nll_loss(y2,tags)
        # 4.nll_loss 处理步骤是 就是把因y2每一行与Label对应的那个值拿出来，再去掉负号，再求均值
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self):
        curr_lr = 0.001
        for epoch in range(20):
            sum_loss = 0
            for i, (imgs, tags) in enumerate(self.train_dataLoader):
                print(tags)

                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                self.net.train()
                y = self.net.forward(imgs)
                print(y.shape, y)
                exit()

                # 交叉熵为loss
                loss = self.loss_fn(y, tags)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.cpu().detach().item()
            avg_loss = sum_loss / len(self.train_dataLoader)
            print("批次：", epoch, "训练集损失：", avg_loss)

            if (epoch + 1) % 20 == 0:  # each 20 epoch, decay the learning rate
                curr_lr /= 3
                for param_group in self.opt.param_groups:
                    param_group['lr'] = curr_lr

        correct = 0
        total = 0
        for images, labels in self.test_dataLoader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('accuracy of the model on the test images: {}%'
              .format(100 * correct / total))

        # if (epoch + 1) % 10 == 0:
        #     sum_score = 0.
        #     test_sum_loss = 0.
        #     for j, (imgs, tags) in enumerate(self.test_dataLoader):
        #         imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
        #         self.net.eval()
        #         test_y = self.net.forward(imgs)
        #
        #         # 交叉熵为loss
        #         test_loss = self.loss_fn(test_y, tags)
        #         test_sum_loss += test_loss.cpu().detach().item()
        #
        #         pred_tags = torch.argmax(test_y, dim=1)
        #         # act_tags = torch.argmax(tags, dim=1)
        #         sum_score += torch.sum(torch.eq(pred_tags, tags).float()).cpu().detach().item()
        #
        #     score = sum_score / len(self.test_dataset)
        #     test_avg_loss = test_sum_loss / len(self.test_dataLoader)
        #     print("批次：", epoch, "测试集损失：", test_avg_loss, "准确率：", score)


# self.summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
# self.summaryWriter.add_scalar("score", score, epoch)

# weight1_layer = self.net.squential[1].weight
# weight2_layer = self.net.squential[6].weight
# weight3_layer = self.net.squential[11].weight
#
# self.summaryWriter.add_histogram("weight1_layer", weight1_layer, epoch)
# self.summaryWriter.add_histogram("weight2_layer", weight2_layer, epoch)
# self.summaryWriter.add_histogram("weight3_layer", weight3_layer, epoch)

# torch.save(self.net.state_dict(), f"./checkpoint/{epoch}.pkl")


# print("y", y)
# print("tags", tags)


if __name__ == '__main__':
    train = Train("./data")
    train()
