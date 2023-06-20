# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import torch.nn as nn
import os
os.chdir(os.getcwd())
import sys
sys.path.append('../../')
from myUtils.dataStep import *
from myUtils.train import *
from config import *


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()


config = Config()


def resnet50():
    return ResNet(block=BottleNeck, layer_list=[2, 2, 2, 2], classNums=5)


if __name__ == "__main__":
    trainLoader, testLoader, data_size = dataLoad(config)
    net = resnet50().to(config.DEVICE)
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    train(net, trainLoader, testLoader, LOSS_FUNC, config, data_size[1], opt='sgd', model_name='resnet50')