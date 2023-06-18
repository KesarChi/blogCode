# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image


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


# diy Dataset for AlexNet, VGG, ResNet
class ImgClassifyDataset(Dataset):
    def __init__(self, data_dir, label, label_map, trans=None):
        self.data_dir = data_dir
        self.label = label
        self.path = os.path.join(self.data_dir, self.label)
        self.img_dir = os.listdir(self.path)
        self.trans = trans
        self.label_map = label_map

    def __getitem__(self, idx):
        img_name = self.img_dir[idx]
        img_item_path = os.path.join(self.data_dir, self.label, img_name)
        img = Image.open(img_item_path)
        if self.trans is not None:
            img = self.trans(img)
        return img, self.label_map[self.label]

    def __len__(self):
        return len(self.img_dir)



