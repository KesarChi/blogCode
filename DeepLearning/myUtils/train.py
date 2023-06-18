# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from .logRecord import *


def train(net, trainLoader, testLoader, LOSS_FUNC, config, test_size, opt='sgd', model_name=None):
    logs = SummaryWriter("logs")
    logg = get_logger(model_name)

    if model_name is not None and os.path.exists(model_name+".pth"):
        net.load_state_dict(torch.load(model_name+".pth"))
        logg.info("Pre-trained parameters loaded: {}".format(model_name+".pth"))
    else:
        logg.info("There is no Pre-trained parameters for Model")

    if opt.upper() == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.LR, momentum=0.9)
    elif opt.upper() == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters())
    else:
        raise KeyError("Your optimizer must be valid")
    logg.info("Optimizer {} loaded".format(opt.upper()))

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
        logs.add_scalar("Accuracy", test_acc / test_size, epoch)
        logs.add_scalar("Test_Loss", test_loss, epoch)
    logs.close()
    logg.info("Acc={}".format(test_acc / test_size))
    torch.save(net.state_dict(), "./"+model_name+".pth")