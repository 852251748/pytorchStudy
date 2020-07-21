import torch
import time
from torch.utils.data import DataLoader
from net import *
from dataset import Mydataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, root, img_size):
        dataset = Mydataset(root, img_size)
        self.dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        torch.cuda.empty_cache()
        if img_size == 12:
            self.net = PNet()
            self.net.load_state_dict(torch.load("../param/test_param/net.pt"))
        elif img_size == 24:
            self.net = RNet()
            # self.net.load_state_dict(torch.load("../param/0_pnet.pt"))
        elif img_size == 48:
            self.net = ONet()
            # self.net.load_state_dict(torch.load("../param/0_pnet.pt"))
        else:
            print("img_size error!", img_size)
            exit()
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.opt = optim.Adam(self.net.parameters())

        self.off_loss_fn = torch.nn.MSELoss()
        self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.summary = SummaryWriter("./logs")

    def __call__(self, num):

        for epoch in range(num):
            trainLoss = 0
            for i, (img, lable) in enumerate(self.dataloader):
                img, lable = img.to(self.device), lable.to(self.device)

                pre = self.net(img)

                if self.img_size == 12:
                    pre = pre.reshape(-1, 15)

                real_conf = lable[:, 0]
                pre_conf = pre[:, 0]

                conf_mask = real_conf < 2

                conf_loss = self.conf_loss_fn(pre_conf[conf_mask], real_conf[conf_mask])

                off_mask = real_conf > 0
                pre_off = pre[off_mask]
                real_off = lable[off_mask]

                off_loss = self.off_loss_fn(torch.tanh_(pre_off[:, 1:5]), real_off[:, 1:5])

                landmask_loss = self.off_loss_fn(pre_off[:, 5:], real_off[:, 5:])

                loss = conf_loss + off_loss + landmask_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                trainLoss += loss.cpu().detach().item()

                # print("i", i, "置信度损失：", conf_loss.cpu().detach().item(), "偏移量损失：", off_loss.cpu().detach().item(),
                #       "五官损失：", landmask_loss.cpu().detach().item())
                if (i + 1) % 50 == 0:
                    print(i)
                    torch.save(self.net.state_dict(), f"../param/test_param/net.pt")

            avgTrainLoss = trainLoss / len(self.dataloader)

            print("批次：", epoch, ",训练集损失：", avgTrainLoss, "time:",
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            self.summary.add_scalar("loss", avgTrainLoss, epoch)
            if epoch % 5 == 0:
                if self.img_size == 12:
                    torch.save(self.net.state_dict(), f"../param/{epoch}_pnet.pt")
                elif self.img_size == 24:
                    torch.save(self.net.state_dict(), f"../param/{epoch}_rnet.pt")
                else:
                    torch.save(self.net.state_dict(), f"../param/{epoch}_onet.pt")


if __name__ == '__main__':
    trainer12 = Train(r"G:\teach_data", 12)
    trainer12(10000)
