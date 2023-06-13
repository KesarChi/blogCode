# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def dataLoad_CIFAR(W, H, batch=32):
    train_set = datasets.CIFAR10("../../data/CIFAR10", train=True,
                                 transform=transforms.Compose([
                                     transforms.Resize((W, H)),
                                     transforms.ToTensor()
                                 ]), download=True)
    train_size = len(train_set)
    train_set = DataLoader(train_set, batch_size=batch, shuffle=True)

    test_set = datasets.CIFAR10("../../data/CIFAR10", train=False,
                                transform=transforms.Compose([
                                    transforms.Resize((W, H)),
                                    transforms.ToTensor()
                                ]))
    test_size = len(test_set)
    test_set = DataLoader(test_set, batch_size=32, shuffle=True)

    return train_set, test_set, [train_size, test_size]
