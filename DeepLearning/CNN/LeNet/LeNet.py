# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import sys
import os
from tqdm import tqdm
os.chdir(os.getcwd())
sys.path.append('../../')
from myUtils.dataStep import *
from config import *

import torch.nn as nn
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
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
    if os.path.exists("./letnet.pth"):
        net.load_state_dict(torch.load('./lenet.pth'))
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = SGD(net.parameters(), config.MAX_LR, config.MOMENTUM)
    logs = SummaryWriter("logs")

    for epoch in tqdm(range(config.EPOCH)):
        running_loss, test_loss, test_acc = 0., 0., 0.
        net.train()
        for data in trainLoader:
            imgs, labels = data
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            out = net(imgs)

            optimizer.zero_grad()

            loss = LOSS_FUNC(out, labels)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
        logs.add_scalar("Train_Loss", running_loss, epoch)

        net.eval()
        with torch.no_grad():
            for data in testLoader:
                imgs, labels = data
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                out = net(imgs)

                loss = LOSS_FUNC(out, labels)
                test_loss += loss.item()
                test_acc += (out.argmax(1) == labels).sum()
        logs.add_scalar("Accuracy", test_acc/data_size[1], epoch)
        logs.add_scalar("Test_Loss", test_loss, epoch)
    logs.close()
    torch.save(net.state_dict(), "lenet.pth")

# type [tensorboard --logdir=logs --port=6007] in Terminal, view train result via browser










