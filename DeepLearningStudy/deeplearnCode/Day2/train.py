from torchvision import datasets, transforms, models
from deeplearnCode.Day2.net import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


class Trainer:
    def __init__(self, root):

        # self.summaryWriter = SummaryWriter("./logs")

        self.train_dataset = datasets.CIFAR10(root, train=True, transform=transforms.ToTensor(),
                                              download=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)

        self.test_dataset = datasets.CIFAR10(root, train=False, transform=transforms.ToTensor(),
                                             download=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=128, shuffle=True)

        self.net = NetV5()
        # self.net = models.resnet18()
        # self.net.load_state_dict(torch.load(r"D:\pycharmworkspace\DeepLearningStudy\deeplearnCode\Day2\param\15_loss0.25884514723135077_accuracy0.8383.pt"))

        self.net.cuda()

        self.opt = torch.optim.Adam(self.net.parameters())

        self.loss_func = torch.nn.CrossEntropyLoss()

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

            grade = 0
            for test_img, test_tag in self.test_dataloader:
                test_img, test_tag = test_img.cuda(), test_tag.cuda()

                pre_y = self.net(test_img)

                grade += (torch.argmax(pre_y, dim=1) == test_tag).sum().float().cpu().detach().item()

            avg_grade = grade / len(self.test_dataset)
            print("批次：", epoch, "损失：", avg_loss, "准确率：", avg_grade)
            if epoch % 5 == 0:
                torch.save(self.net.state_dict(), f"./param/{epoch}_loss{avg_loss}_accuracy{avg_grade}.pt")
            # self.summaryWriter.add_scalar("loss", avg_loss, epoch)
            # self.summaryWriter.add_scalar("grade", avg_grade, epoch)


if __name__ == '__main__':
    train = Trainer("D:\Alldata")
    train(100)
