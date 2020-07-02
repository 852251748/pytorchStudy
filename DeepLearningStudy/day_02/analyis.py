from torch.utils.tensorboard import SummaryWriter
import cv2
from day_02.net import *

if __name__ == '__main__':

    net = NetV2()
    # net.load_state_dict(torch.load("./checkpoint/10.pkl"))
    print(net.squential)
    exit()
    summary = SummaryWriter("./logs")

    weight1_layer = net.squential[0].weight
    weight2_layer = net.squential[4].weight
    weight3_layer = net.squential[8].weight

    summary.add_histogram("weight1_layer", weight1_layer)
    summary.add_histogram("weight2_layer", weight2_layer)
    summary.add_histogram("weight3_layer", weight3_layer)
    #
    # cv2.waitKey(0)
    # net1 = NetV1()
    # print(net1.squential)
    #
    # weight1_layer = net.squential[0].weight
    # weight2_layer = net.squential[2].weight
    # weight3_layer = net.squential[4].weight
    #
    # summary.add_histogram("weight1_layer", weight1_layer)
    # summary.add_histogram("weight2_layer", weight2_layer)
    # summary.add_histogram("weight3_layer", weight3_layer)