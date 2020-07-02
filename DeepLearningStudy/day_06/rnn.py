from torchvision import datasets, transforms
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional  as F

DEVICE = "cuda:0"


class MyRnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(28 * 1, 128, 2, batch_first=True, bidirectional=True)
        self.output = nn.Linear(128 * 2, 10)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3)
        input = x.reshape(n, h, -1)
        h0 = torch.zeros(2 * 2, n, 128).to(DEVICE)
        c0 = torch.zeros(2 * 2, n, 128).to(DEVICE)

        output, (hn, cn) = self.rnn(input, (h0, c0))  # hn最后一次输出，output最后一次输出

        outputs = self.output(output[:, -1, :])  # 需要最后一步的数据
        return outputs


class MyRnn2(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnnCell1 = nn.GRUCell(28, 128)
        self.rnnCell2 = nn.GRUCell(128, 128)
        self.outLayer = nn.Linear(128, 10)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(n, h, w * c)

        h0 = torch.zeros(n, 128).to(DEVICE)
        h1 = torch.zeros(n, 128).to(DEVICE)
        for i in range(h):
            h0 = F.relu(self.rnnCell1(x[:, i, :], h1))
            h1 = self.rnnCell2(h0)
        outs = self.outLayer(h1)
        return outs


if __name__ == '__main__':
    # myRnn = MyRnn()
    myRnn = MyRnn2()
    myRnn.to(DEVICE)
    opt = optim.Adam(myRnn.parameters())

    lossFunc = nn.CrossEntropyLoss()
    # myRnn.to(DEVICE)
    trainDataset = datasets.MNIST("../day_01/data", train=True, transform=transforms.ToTensor())
    testDataset = datasets.MNIST("../day_01/data", train=False, transform=transforms.ToTensor())

    trainDataloader = DataLoader(trainDataset, batch_size=100, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=100, shuffle=True)

    for epoch in range(100000):
        trainLoss = 0
        testLoss = 0
        for data, lable in trainDataloader:
            myRnn.train()
            data, lable = data.to(DEVICE), lable.to(DEVICE)

            predict = myRnn(data)

            loss = lossFunc(predict, lable)

            opt.zero_grad()
            loss.backward()
            opt.step()

            trainLoss += loss.detach().item()
        score = 0
        for dataT, lableT in testDataloader:
            dataT, lableT = dataT.to(DEVICE), lableT.to(DEVICE)
            myRnn.eval()
            pre = myRnn(dataT)

            loss = lossFunc(pre, lableT)

            preLable = torch.argmax(pre, dim=1)
            score += torch.sum(torch.eq(lableT, preLable).float()).detach().item()

            testLoss += loss.detach().item()
        avgLossTest = testLoss / len(testDataloader)
        avgLossTrain = trainLoss / len(trainDataloader)
        score = score / len(testDataset)
        print("训练集损失：", avgLossTrain, "测试集损失：", avgLossTest, "准确率：", score)
