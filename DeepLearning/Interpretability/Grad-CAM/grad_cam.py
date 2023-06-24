# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.
import os

import PIL.Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
from PIL import Image
from DeepLearning.myUtils.tools import *


gradients = None
activations = None


def hook_forward(module, ac_i, ac_o):
    global activations
    activations = ac_o


def hook_backward(module, grad_i, grad_o):
    global gradients
    gradients = grad_o


def generate_heatmap():
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= torch.mean(gradients[0], dim=[0, 2, 3])[i]
    heatmap = F.relu(torch.mean(activations, dim=1).squeeze())
    heatmap /= torch.max(heatmap)
    return heatmap


def grad_plot(img_tensor, heatmap, shape, save_file):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(to_pil_image(img_tensor, mode='RGB'))
    heat = to_pil_image(heatmap.detach(), mode='F').resize(shape, resample=PIL.Image.BICUBIC)
    cmap = colormaps['jet']
    heat = (224 * cmap(np.asarray(heat) ** 2)[:, :, :3]).astype('uint8')
    ax.imshow(heat, alpha=.4, interpolation='nearest')
    plt.savefig(save_file)
    plt.show()


def load_img(path, conf):
    img = Image.open(path).convert('RGB')
    img = conf.data_trans(img)
    img = img.to(conf.DEVICE)
    return img


def record_grads(net, LOSS_FUNC, img, label):
    h1 = net.block4[-1].register_forward_hook(hook_forward)
    h2 = net.block4[-1].register_full_backward_hook(hook_backward)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    out = net(img)
    print(out)
    loss = LOSS_FUNC(out, label.long())
    loss.backward()
    h1.remove()
    h2.remove()


def single_grad_visualization(net, LOSS_FUNC, img_path, label, config, shape=(224, 224), save_file='./1.png'):
    img = load_img(img_path, config)
    label = torch.Tensor(label).to(config.DEVICE)
    record_grads(net, LOSS_FUNC, img, label)
    heatmap = generate_heatmap()
    grad_plot(img, heatmap, shape, save_file)


def multi_grad_visualization(net, LOSS_FUNC, img_path_list, label, config, shape=(224,224), save_path='./'):
    filenames = []
    # print(labels)
    labels = [[la] for la in label]

    if type(img_path_list) == str:
        filenames = get_all_picnames_of_path(img_path_list)
    elif type(img_path_list) == list:
        filenames = img_path_list

    for i in range(len(filenames)):
        global activations, gradients
        activations, gradients = None, None
        single_grad_visualization(net, LOSS_FUNC, os.path.join(img_path_list, filenames[i]), labels[i], config, shape=shape, save_file=os.path.join(save_path, filenames[i]))
        # img = load_img(os.path.join(img_path_list,filenames[i]), config)
        # record_grads(net, LOSS_FUNC, img, labels[i])
        # heatmap = generate_heatmap()
        # grad_plot(img, heatmap, shape, './'+filenames[i])

















