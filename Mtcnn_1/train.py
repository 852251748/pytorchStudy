from data import *
from net import *
from torch.utils.data import DataLoader
from torch import optim
import torch
# from torch.utils.tensorboard import SummaryWriter
DEVICE = "cuda:0"


class Train:
    def __init__(self, root, imagesize):
        self.imgsize = imagesize

        if self.imgsize == 12:
            self.net = PNet()
            self.net.load_state_dict(torch.load("./param/pnet.pt"))
        elif self.imgsize == 24:
            self.net = RNet()
            self.net.load_state_dict(torch.load("./param/rnet.pt"))
        else:
            self.net = ONet()
            self.net.load_state_dict(torch.load("./param/onet.pt"))

        self.net.to(DEVICE)

        self.dataset = Mydataset(root, imagesize)
        self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=True)

        self.opt = optim.Adam(self.net.parameters(), lr=0.001)

    def __call__(self, epoch):
        for i in range(epoch):
            sumLoss = 0
            # last_loss = 10
            for j, (img, lable) in enumerate(self.dataloader):
                img, lable = img.to(DEVICE), lable.to(DEVICE)

                predict = self.net(img)

                if self.imgsize == 12:
                    predict = predict.reshape(-1, 5)

                torch.sigmoid_(predict[:, 0])

                # 置信度损失
                mask = lable[:, 0] < 2
                indexC = mask.nonzero()[:, 0]
                preCond = predict[indexC][:, 0]
                actCond = lable[indexC][:, 0]
                condLoss = torch.mean((preCond - actCond) ** 2)

                # 标注框偏移量损失
                mask = lable[:, 0] > 0
                indexO = mask.nonzero()[:, 0]
                preBoxOf = predict[indexO][:, 1:5]
                actBoxOf = lable[indexO][:, 1:5]
                boxLoss = torch.mean((preBoxOf - actBoxOf) ** 2)

                #只有O网络训练五官偏移量
                # 计算五官偏移量的损失
                ofLdMLoss = 0
                if self.imgsize == 48:
                    ofLdMPre = predict[indexO][:, 5:]
                    ofLdMAct = lable[indexO][:, 5:]
                    ofLdMLoss = torch.mean((ofLdMPre - ofLdMAct) ** 2)

                loss = condLoss + boxLoss + ofLdMLoss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sumLoss += loss.cpu().detach().item()
                print("批次：", j, "损失：", loss.detach().item(),condLoss.detach().item() , boxLoss.detach().item() , ofLdMLoss)
            avg_loss = sumLoss / len(self.dataloader)

            print("批次：", i, "损失：", avg_loss)
            # if last_loss > avg_loss:
            #     last_loss = avg_loss
            if self.imgsize == 12:
                torch.save(self.net.state_dict(), "./param/pnet.pt")
            elif self.imgsize == 24:
                torch.save(self.net.state_dict(), "./param/rnet.pt")
            else:
                torch.save(self.net.state_dict(), "./param/onet.pt")

#
if __name__ == '__main__':
    train = Train("E:\mtcnn_data", 12)
    train(100)
#     # train = Train("D:\celeba_3", 24)
#     # train(10000)
