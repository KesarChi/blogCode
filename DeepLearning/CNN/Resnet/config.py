# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import torch
from torchvision import transforms


class Config:
    def __init__(self):
        self.data_path = "../../../data/Flower"
        self.EPOCH = 30
        self.LR = 1e-3
        self.DEVICE = torch.device('cuda')
        self.data_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.label_map = {'Lilly': 0, 'Lotus': 1, 'Orchid': 2, 'Sunflower': 3, 'Tulip': 4}