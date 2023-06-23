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
    def __init__(self, in_channel, out_channel, stride=1, ratio=4, downsample=False):
        super(BottleNeck, self).__init__()
        self.ratio = ratio
        self.down = downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.ratio, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channel * self.ratio),
        )
        self.down_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel * self.ratio, kernel_size=1, stride=stride,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channel * self.ratio)
        )

    def forward(self, x):
        raw = x
        raw = self.down_shortcut(raw) if self.down else raw
        out = self.model(x)
        return self.relu(raw + out)


class ResNet(nn.Module):
    def __init__(self, layerNums, classNums):
        super(ResNet, self).__init__()
        self._in_c = 64
        self.ratio = 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block1 = self._stack_block(channel=64, stride=1, nums=layerNums[0])
        self.block2 = self._stack_block(channel=128, stride=2, nums=layerNums[1])
        self.block3 = self._stack_block(channel=256, stride=2, nums=layerNums[2])
        self.block4 = self._stack_block(channel=512, stride=2, nums=layerNums[3])
        self.model = nn.Sequential(
            self.conv,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512 * self.ratio, out_features=classNums)
        )

    def _stack_block(self, channel, stride, nums):
        blocks = [BottleNeck(in_channel=self._in_c, out_channel=channel, stride=stride, downsample=True)]
        for i in range(1, nums):
            blocks.append(BottleNeck(in_channel=channel*4, out_channel=channel, stride=1))
        self._in_c = channel * self.ratio
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


config = Config()


if __name__ == "__main__":
    trainLoader, testLoader, data_size = dataLoad(config)
    net = ResNet([3, 4, 6, 3], 5).to(config.DEVICE)
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    train(net, trainLoader, testLoader, LOSS_FUNC, config, data_size[1], opt='sgd', model_name='resnet50')

    # x = torch.randn(4, 3, 224, 224)
    # net = ResNet([3,4,6,3], 5)
    # out = net(x)
    # print(out)
