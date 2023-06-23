# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from LeNet import *
from config import *
import os
os.chdir(os.getcwd())
from DeepLearning.myUtils.dataStep import *
from DeepLearning.myUtils.train import *


config = Config()


if __name__ == "__main__":
    trainLoader, testLoader, data_size = dataLoad_CIFAR(32, 32)
    net = LeNet().to(config.DEVICE)
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    train(net, trainLoader, testLoader, LOSS_FUNC, config, data_size[1], opt='sgd', model_name='lenet')

    # type [tensorboard --logdir=logs --port=6007] in Terminal, view train result via browser