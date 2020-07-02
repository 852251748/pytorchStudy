from torch.utils.data import DataLoader
from day_07.data import *
from day_07.net import *
from torch import optim
import torch
from torchvision import utils


class Trainer:
    def __init__(self, root):
        self.dataset = MyFaceData(root)
        self.datasetloader = DataLoader(self.dataset, batch_size=100, shuffle=True)

        self.net = GANNet()
        self.net.cuda()

        self.optD = optim.Adam(self.net.dnet.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.optG = optim.Adam(self.net.gnet.parameters(), lr=0.0001, betas=(0.5, 0.9))

    def __call__(self):
        for epoch in range(10000):
            for img in self.datasetloader:
                img = img.cuda()

                noiseg = torch.normal(0, 1, (100, 128, 1, 1)).cuda()
                noised = torch.normal(0, 1, (100, 128, 1, 1)).cuda()

                lossD = self.net.GetDLoss(noised, img)

                self.optD.zero_grad()
                lossD.backward()
                self.optD.step()

                lossG = self.net.GetGLoss(noiseg)

                self.optG.zero_grad()
                lossG.backward()
                self.optG.step()

                print("生成器的损失：", lossG.cpu().detach().item(), "判别器的损失：", lossD.cpu().detach().item())
            noisep = torch.normal(0, 1, (8, 128, 1, 1)).cuda()
            img = self.net(noisep).cpu().detach()
            utils.save_image(img, "newPic.jpg", normalize=True, range=(-1, 1))


if __name__ == '__main__':
    trainer = Trainer(r"E:\BaiduNetdiskDownload\DeepLearingDownLoad\2020-04-24GAN\newCode\Cartoon_faces\faces")
    trainer()
