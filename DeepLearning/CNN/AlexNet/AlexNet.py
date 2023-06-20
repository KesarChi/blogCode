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


class AlexNet(nn.Module):
    def __init__(self, classNums):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=classNums)
        )

    def forward(self, x):
        return self.model(x)


config = Config()


if __name__ == "__main__":
    trainLoader, testLoader, data_size = dataLoad(config)
    net = AlexNet(classNums=5).to(config.DEVICE)
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    train(net, trainLoader, testLoader, LOSS_FUNC, config, data_size[1], opt='sgd', model_name='alexnet')

# type [tensorboard --logdir=logs --port=6007] in Terminal, view train result via browser


