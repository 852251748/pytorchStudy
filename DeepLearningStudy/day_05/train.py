from day_05.data import *
# from day_05.net import *
from day_05.net_new import *
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda:0"


class Train:
    def __init__(self, root, imagesize):

        self.summary = SummaryWriter("./logs")
        self.imgsize = imagesize
        # 训练数据集
        self.trainDataset = Mydataset(root, self.imgsize)
        self.traindataloder = DataLoader(self.trainDataset, batch_size=512, shuffle=True)

        self.claLoss = nn.BCELoss()
        self.boxLoss = nn.MSELoss()

        # 创建网络实例
        if self.imgsize == 12:
            self.net = PNet()
            # self.net.load_state_dict(torch.load("./pnet.pt"))
        elif self.imgsize == 24:
            self.net = RNet()
            # self.net.load_state_dict(torch.load("./rnet.pt"))
        else:
            self.net = ONet()
            # self.net.load_state_dict(torch.load("./onet.pt"))

        self.net.to(DEVICE)

        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, epoch):
        for i in range(epoch):
            trainLoss = 0
            for j, (image, cond, boxof, ldmof) in enumerate(self.traindataloder):
                self.net.train()
                image, cond, boxof, ldmof = image.to(DEVICE), cond.to(DEVICE), boxof.to(DEVICE), ldmof.to(DEVICE)

                # preCond, preBoxof, preLdmof = self.net(image)

                preCond, preBoxof = self.net(image)
                if self.imgsize == 12:
                    preCond = preCond.reshape(-1, 1)
                    preBoxof = preBoxof.reshape(-1, 4)
                    # preLdmof = preLdmof.reshape(-1, 10)

                # 计算置信度损失
                cMask = cond < 2
                cCondPre = preCond[cMask]
                cCondAct = cond[cMask]
                cCondLoss = self.claLoss(cCondPre, cCondAct)  # 二分类交叉熵函数

                # 计算标注框偏移量的损失
                ofMask = cond[:, 0] > 0
                ofPreIndex = ofMask.nonzero()[:, 0]
                ofBoxPre = preBoxof[ofPreIndex]
                ofBoxAct = boxof[ofPreIndex]
                # ofBoxLoss = torch.mean((ofBoxPre - ofBoxAct) ** 2)
                ofBoxLoss = self.boxLoss(ofBoxPre, ofBoxAct)

                # 计算五官偏移量的损失
                # ofLdMPre = preLdmof[ofPreIndex]
                # ofLdMAct = ldmof[ofPreIndex]
                # ofLdMLoss = torch.mean((ofLdMPre - ofLdMAct) ** 2)

                # loss = cCondLoss + ofBoxLoss + ofLdMLoss
                loss = cCondLoss + ofBoxLoss
                # print( loss.requires_grad)
                # exit()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                trainLoss += loss.cpu().detach().item()
                # print(j, "置信度损失：", cCondLoss.cpu().detach().item(), "标注框损失：", ofBoxLoss.cpu().detach().item())
                # "五官损失：",
                # ofLdMLoss.cpu().detach().item())
            # #pnet
            # weight1 = self.net.sequential[0].weight
            # weight2 = self.net.sequential[4].weight
            # weight3 = self.net.sequential[7].weight

            # rnet
            # weight1 = self.net.sequential[0].weight
            # weight2 = self.net.sequential[4].weight
            # weight3 = self.net.sequential[8].weight

            avgTrainLoss = trainLoss / len(self.traindataloder)
            self.summary.add_scalar("train_loss", avgTrainLoss, i)

            # self.summary.add_histogram("weight1", weight1, i)
            # self.summary.add_histogram("weight2", weight2, i)
            # self.summary.add_histogram("weight3", weight3, i)

            print("批次：", i, ",训练集损失：", avgTrainLoss)

            if self.imgsize == 12:
                torch.save(self.net.state_dict(), "pnet.pt")
            elif self.imgsize == 24:
                torch.save(self.net.state_dict(), "rnet.pt")
            else:
                torch.save(self.net.state_dict(), "onet.pt")


if __name__ == '__main__':
    train = Train("D:/pycharm_workspace/mtcnn_data", 48)
    train(1000000)
