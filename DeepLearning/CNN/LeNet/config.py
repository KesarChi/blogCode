# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import torch


class Config:
    def __init__(self):
        self.DEVICE = torch.device('cuda')
        self.EPOCH = 20
        self.LR = 1e-3
        self.MOMENTUM = 0.9

