# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import os
from DeepLearning.CNN.Resnet.ResNet import ResNet
from DeepLearning.CNN.Resnet.config import *
from DeepLearning.myUtils.train import *
os.chdir(os.getcwd())


config = Config()


if __name__ == "__main__":
    net = ResNet([3, 4, 6, 3], 5).to(config.DEVICE)
    net.load_state_dict(torch.load("../../CNN/ResNet/resnet50.pth"))
    net.eval()
    print(net.block4)






