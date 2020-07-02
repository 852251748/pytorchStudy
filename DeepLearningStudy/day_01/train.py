# 训练模型
from torch import optim
from day_01.data import *
from day_01.net import *
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
import torch

DEVICE = "cuda:0"


class Train:
    def __init__(self, root):
        # 创建tensorboard数据记录目录
        self.summaryWriter = SummaryWriter("./logs")
        # 加载训练数据集
        self.train_dataset = MNISTDataset(root, True)
        # 将60000份数据分成每批次100个数据
        self.train_dataLoader = dataloader.DataLoader(self.train_dataset, batch_size=100, shuffle=True)

        # 加载测试数据集
        self.test_dataset = MNISTDataset(root, True)
        # 将10000份数据分成每批次100个数据
        self.test_dataLoader = dataloader.DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建模型
        self.net = NetV3()
        # 加载保存的参数
        # self.net.load_state_dict(torch.load("./checkpoint/11.pkl"))
        self.net.cuda()

        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        for epoch in range(100000):
            sum_loss = 0
            for i, (imgs, tags) in enumerate(self.train_dataLoader):
                imgs, tags = imgs.cuda(), tags.cuda()
                self.net.train()
                y = self.net.forward(imgs)

                loss = torch.mean((tags - y) ** 2)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.cpu().detach().item()

            sum_score = 0.
            test_sum_loss = 0.
            for j, (imgs, tags) in enumerate(self.test_dataLoader):
                imgs, tags = imgs.cuda(), tags.cuda()
                self.net.eval()
                test_y = self.net.forward(imgs)

                test_loss = torch.mean((tags - test_y) ** 2)
                test_sum_loss += test_loss.cpu().detach().item()

                pred_tags = torch.argmax(test_y, dim=1)
                act_tags = torch.argmax(tags, dim=1)
                sum_score += torch.sum(torch.eq(pred_tags, act_tags).float()).cpu().detach().item()

            score = sum_score / len(self.test_dataset)
            test_avg_loss = test_sum_loss / len(self.test_dataLoader)
            avg_loss = sum_loss / len(self.train_dataLoader)

            self.summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
            self.summaryWriter.add_scalar("score", score, epoch)

            # torch.save(self.net.state_dict(), f"./checkpoint/{epoch}.pkl")
            print(epoch, avg_loss, test_avg_loss, score)

            # print("y", y)
            # print("tags", tags)


if __name__ == '__main__':
    train = Train("data/MNIST_IMG")
    train()
