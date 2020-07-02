import torch
from day_01 import net

v3net = net.NetV3()
weight1_layer = v3net.sequential[2].weight
print(weight1_layer.shape,weight1_layer)