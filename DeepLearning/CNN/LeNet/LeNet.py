# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import sys
import os

os.chdir(os.getcwd())
sys.path.append('../../')
from myUtils.dataStep import *
from myUtils.train import *
from config import *
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)


config = Config()

if __name__ == "__main__":
    trainLoader, testLoader, data_size = dataLoad_CIFAR(32, 32)
    net = LeNet().to(config.DEVICE)
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    train(net, trainLoader, testLoader, LOSS_FUNC, config, data_size[1], opt='sgd', model_name='lenet')

# type [tensorboard --logdir=logs --port=6007] in Terminal, view train result via browser
