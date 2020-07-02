import torch
from day_06.data import *
from day_06.net import *
from torch.utils.data import DataLoader
from torch import optim

DEVICE = "cuda:0"


class Trainer:
    def __init__(self):
        self.dataset = MyData("./code")
        self.dataLoader = DataLoader(self.dataset, batch_size=100, shuffle=True)
        self.net = Cnn2Seq()
        self.net.to(DEVICE)
        self.opt = optim.Adam(self.net.parameters())
        self.lossFun = nn.MSELoss()

    def __call__(self):

        for epoch in range(10000):
            score = 0
            sumLoss = 0
            for i, (img, lable) in enumerate(self.dataLoader):
                img, lable = img.to(DEVICE), lable.to(DEVICE)
                pre = self.net(img)

                loss = self.lossFun(pre, lable)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                preY = torch.argmax(pre, dim=2)
                lableY = torch.argmax(lable, dim=2)

                score += torch.sum(torch.eq(preY, lableY).float()).cpu().detach().item()
                sumLoss += loss.cpu().detach().item()
                if i % 20 == 0:
                    print(lableY[0].cpu().detach().numpy(), preY[0].cpu().detach().numpy())
            avgloss = sumLoss / len(self.dataLoader)
            avgscore = score / (len(self.dataset) * 4)
            print("损失：", avgloss, "精度：", avgscore)
            # exit()

            # print("损失：", loss)


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
