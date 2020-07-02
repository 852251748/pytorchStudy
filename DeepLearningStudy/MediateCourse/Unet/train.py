import torch
from MediateCourse.Unet.net import MainNet
from MediateCourse.Unet.dataset import Mydataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision.utils import save_image
import os


class Train:
    def __init__(self, root):
        self.dataset = Mydataset(root)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.BCELoss()
        self.net = MainNet()
        self.net.to(self.device)
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        img_save_path = r"./preImg"

        for epoch in range(100000):
            for i, (img, label) in enumerate(self.dataloader):
                img, label = img.to(self.device), label.to(self.device)

                pre = self.net(img)

                loss = self.loss_fn(pre, label)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % 20 == 0:
                    print("epoch:", epoch, "loss:", loss.cpu().detach().item())
                    torch.save(self.net.state_dict(), f"./param/{i}.pkl")
                    img1 = img[0]
                    label1 = label[0]
                    pre1 = pre[0]

                    pic = torch.stack([img1, label1, pre1], dim=0)

                    save_image(pic.cpu(), os.path.join(img_save_path, '{}.png'.format(i)))


if __name__ == '__main__':
    trainer = Train(r"D:\Alldata\VOCdevkit\VOC2012")
    trainer()
