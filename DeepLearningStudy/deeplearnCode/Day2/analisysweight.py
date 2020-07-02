from deeplearnCode.Day2.net import *
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    net = NetV2()
    net.load_state_dict(torch.load("./param/16.pt"))
    print(net)

    summaryWriter = SummaryWriter("./logs")
    layer1_weight = net.seq[0].weight
    layer2_weight = net.seq[2].weight
    layer3_weight = net.seq[5].weight
    layer4_weight = net.seq[7].weight
    layer5_weight = net.seq[10].weight
    layer6_weight = net.seq[12].weight

    summaryWriter.add_histogram("layer1_weight", layer1_weight)
    summaryWriter.add_histogram("layer2_weight", layer2_weight)
    summaryWriter.add_histogram("layer3_weight", layer3_weight)
    summaryWriter.add_histogram("layer4_weight", layer4_weight)
    summaryWriter.add_histogram("layer5_weight", layer5_weight)
    summaryWriter.add_histogram("layer6_weight", layer6_weight)
