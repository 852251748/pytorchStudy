from day_05.data import *
from day_05.net import *
from torch.utils.data import DataLoader
from torch import optim
import torch

DEVICE = "cuda:0"


class Train:
    def __init__(self, root, imagesize):
        self.imgsize = imagesize
        # 训练数据集
        self.trainDataset = Mydataset(root, self.imgsize)
        self.traindataloder = DataLoader(self.trainDataset, batch_size=30, shuffle=True)

        # 测试训练集
        self.testDataset = Mydataset(root, self.imgsize, True)
        self.testDataloder = DataLoader(self.testDataset, batch_size=30, shuffle=True)

        # 创建网络实例
        if self.imgsize == 12:
            self.net = PNet()
        elif self.imgsize == 24:
            self.net = RNet()
        else:
            self.net = ONet()

        self.net.to(DEVICE)

        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, epoch):
        for i in range(epoch):
            trainLoss = 0
            for j, (image, cond, ofset) in enumerate(self.traindataloder):
                self.net.train()
                image, cond, ofset = image.to(DEVICE), cond.to(DEVICE), ofset.to(DEVICE)

                preCond, preOfset = self.net(image)

                if self.imgsize == 12:
                    preCond = preCond.reshape(-1, 1)
                    preBoxof = preOfset.reshape(-1, 4)

                # 计算置信度损失
                cMask = cond[:, ] < 2
                cCondPre = preCond[cMask]
                cCondAct = cond[cMask]
                cCondLoss = torch.mean((cCondPre[:, ] - cCondAct[:, ]) ** 2)

                # 计算标注框偏移量的损失
                ofMask = cond[:, 0] > 0
                ofPreIndex = torch.nonzero(ofMask)[:, 0]
                ofBoxPre = preOfset[ofPreIndex]
                ofBoxAct = ofset[ofPreIndex]
                ofBoxLoss = torch.mean((ofBoxPre[:, ] - ofBoxAct[:, ]) ** 2)

                loss = cCondLoss + ofBoxLoss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # if np.isnan(ofBoxLoss.cpu().detach()):
                #     print("test", j, ofPreIndex, ofBoxPre, ofBoxAct,cond)
                # print(j, loss, cCondLoss, ofBoxLoss, ofLdMLoss)
                trainLoss += loss.cpu().detach().item()

            # 测试模型
            testLoss = 0
            sumScore = 0
            for k, (img, cond, boxof) in enumerate(self.testDataloder):
                self.net.eval()
                img, cond, boxof = img.to(DEVICE), cond.to(DEVICE), boxof.to(DEVICE)

                preCond, preBoxof = self.net(img)
                # 计算置信度损失
                cMask = cond[:, ] < 2
                cCondPre = preCond[cMask]
                cCondAct = cond[cMask]
                cCondLoss = torch.mean((cCondPre[:, ] - cCondAct[:, ]) ** 2)

                # 计算标注框偏移量的损失
                ofMask = cond[:, 0] > 0
                ofPreIndex = torch.nonzero(ofMask)[:, 0]
                ofBoxPre = preBoxof[ofPreIndex]
                ofBoxAct = boxof[ofPreIndex]
                ofBoxLoss = torch.mean((ofBoxPre[:, ] - ofBoxAct[:, ]) ** 2)

                # # 计算五官偏移量的损失
                # ofLdMPre = preLdmof[ofPreIndex]
                # ofLdMAct = ldmof[ofPreIndex]
                # ofLdMLoss = torch.mean((ofLdMPre[:, ] - ofLdMAct[:, ]) ** 2)

                # maskAc = preCond[:,]>0.65
                loss = (cCondLoss + ofBoxLoss)

                testLoss += loss.cpu().detach().item()
                # sumScore += torch.sum(torch.eq(testPre[:, 0], tag[:, 0]).float()).cpu().detach().item()

            avgTrainLoss = trainLoss / len(self.traindataloder)
            avgTestLoss = testLoss / len(self.testDataloder)
            score = sumScore / len(self.testDataset)
            print("批次：", i, ",训练集损失：", avgTrainLoss, ",测试集损失：", avgTestLoss, ",准确率：", score)
            # print("批次：", i, ",训练集损失：", avgTrainLoss)

            if self.imgsize == 12:
                torch.save(self.net.state_dict(), "pnet")
            elif self.imgsize == 24:
                torch.save(self.net.state_dict(), "rnet")
            else:
                torch.save(self.net.state_dict(), "onet")


if __name__ == '__main__':
    train = Train("D:/pycharm_workspace/mtcnn_data", 12)
    train(100)
