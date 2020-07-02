# 训练模型
from torch import optim
from work.data import *
from work.net import *
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from work.CenterLoss import *
import torch.optim.lr_scheduler as lr_scheduler

DEVICE = "cuda:0"

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def decet(feature, targets, epoch, save_path):
    color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # plt.ion()
    # plt.clf()
    for j in cls:
        mask = [targets == j]
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right")  # 如果写在plot上面，则标签内容不能显示完整
        plt.title("epoch={}".format(str(epoch)))

    plt.savefig('{}/{}.jpg'.format(save_path, epoch + 1))
    # plt.draw()
    # plt.pause(0.001)


class Train:
    def __init__(self, root, lamda):
        # 创建tensorboard数据记录目录
        self.summaryWriter = SummaryWriter("./logs")
        # 加载训练数据集
        self.train_dataset = MNISTDataset(root, True)
        # 将60000份数据分成每批次100个数据
        self.train_dataLoader = dataloader.DataLoader(self.train_dataset, batch_size=5000, shuffle=True)

        # # 加载测试数据集
        # self.test_dataset = MNISTDataset(root, False)
        # # 将10000份数据分成每批次100个数据
        # self.test_dataLoader = dataloader.DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建模型
        self.net = NetV4()
        # 加载保存的参数
        self.net.load_state_dict(torch.load("./param/160.pkl"))
        self.net.cuda()

        self.center_loss = CenterLoss(10, 2, lamda).cuda()
        self.ec = nn.CrossEntropyLoss()

        # 创建优化器
        self.output_opt = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.center_opt = optim.SGD(self.center_loss.parameters(), lr=0.5)
        # self.output_opt = optim.Adam(self.net.parameters(), lr=0.001)
        # self.center_opt = optim.Adam(self.center_loss.parameters(), lr=0.5)
        self.scheduler = lr_scheduler.StepLR(self.output_opt, 20, gamma=0.8)

    def __call__(self):
        count = 0
        for epoch in range(100000):
            self.scheduler.step(epoch)
            sum_loss = 0
            feat = []
            target = []
            for i, (imgs, tags) in enumerate(self.train_dataLoader):
                imgs, tags = imgs.cuda(), tags.cuda()

                self.net.train()
                feature, y = self.net(imgs)

                centerloss = self.center_loss(tags, feature)
                ce_loss = self.ec(y, tags)

                loss = 0.5 * centerloss + ce_loss

                self.center_opt.zero_grad()
                self.output_opt.zero_grad()
                loss.backward()
                self.center_opt.step()
                self.output_opt.step()

                count += 1

                feat.append(feature)
                target.append(tags)
                sum_loss += loss.cpu().detach().item()
                # print(i, loss.cpu().detach().item(), centerloss.cpu().detach().item(), ce_loss.cpu().detach().item())

            features = torch.cat(feat, 0)
            targets = torch.cat(target, 0)
            # print(features.shape, targets.shape)
            decet(features.data.cpu(), targets.data.cpu(), epoch, "./resultsPic")
            torch.save(self.net.state_dict(), f"./param/{epoch}.pkl")
            print(epoch, (sum_loss / len(self.train_dataLoader)))


if __name__ == '__main__':
    train = Train("data/MNIST_IMG", 0.003)
    train()
