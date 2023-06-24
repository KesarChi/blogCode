# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import torch.nn as nn
from DeepLearning.CNN.Resnet.ResNet import ResNet
from DeepLearning.CNN.Resnet.config import *
from grad_cam import *
os.chdir(os.getcwd())


config = Config()
img_path = "../../../data/grad-cam/T.jpg"
img_list = "../../../data/grad-cam/"
labels = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]


if __name__ == "__main__":
    net = ResNet([3, 4, 6, 3], 5).to(config.DEVICE)
    net.load_state_dict(torch.load("../../CNN/ResNet/resnet50.pth"))
    LOSS_FUNC = nn.CrossEntropyLoss().to(config.DEVICE)

    # single_grad_visualization(net, LOSS_FUNC, img_path, [0], config)

    multi_grad_visualization(net, LOSS_FUNC, img_list, labels, config, shape=(224,224), save_path='./save/')







