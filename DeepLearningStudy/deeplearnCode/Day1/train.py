import torch
from deeplearnCode.Day1.data import *
from deeplearnCode.Day1.net import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, root):
        self.root = root
        self.summaryWriter = SummaryWriter("./logs")

        self.train_dataset = MINISTDataset(root, True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)

        self.test_dataset = MINISTDataset(root, False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=128, shuffle=True)

        self.net = MlpNet()

        self.net.cuda()

        self.opt = torch.optim.Adam(self.net.parameters())

        self.loss_func = torch.nn.MSELoss()

    def __call__(self, num):
        for epoch in range(num):
            sum_loss = 0
            for j, (img, tag) in enumerate(self.train_dataloader):
                img, tag = img.cuda(), tag.cuda()

                y = self.net(img)

                loss = self.loss_func(y, tag)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.cpu().detach().item()

            avg_loss = sum_loss / len(self.train_dataloader)

            torch.save(self.net.state_dict(), f"./param/{epoch}.pt")
            grade = 0
            for test_img, test_tag in self.test_dataloader:
                test_img, test_tag = test_img.cuda(), test_tag.cuda()

                pre_y = self.net(test_img)

                grade += (torch.argmax(pre_y, dim=1) == torch.argmax(test_tag,
                                                                     dim=1)).sum().float().cpu().detach().item()

            avg_grade = grade / len(self.test_dataset)
            print("损失：", avg_loss, "准确率：", avg_grade)
            self.summaryWriter.add_scalar("loss", avg_loss, epoch)
            self.summaryWriter.add_scalar("grade", avg_grade, epoch)


if __name__ == '__main__':
    train = Trainer("../../day_01/data/MNIST_IMG")
    train(20)
