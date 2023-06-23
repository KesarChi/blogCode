# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from ResNet import *
from config import *
import os
os.chdir(os.getcwd())
from DeepLearning.myUtils.dataStep import *
from DeepLearning.myUtils.train import *


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