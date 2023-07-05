# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.
import numpy as np

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Resize
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os
os.chdir(os.getcwd())


trans = Compose([Resize((224, 224)),
                 ToTensor()])


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
    input_tensors = Image.open("/mnt/m/code/blogCode/data/grad-cam/T.jpg").convert('RGB')
    input_tensors = input_tensors.resize((224, 224))

    input_tensor = trans(input_tensors)
    input_tensor = input_tensor.unsqueeze(0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(224)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    input_tensors = np.array(input_tensors) / 255
    vis = show_cam_on_image(input_tensors, grayscale_cam, use_rgb=True)
    res = Image.fromarray(vis)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(to_pil_image(vis, mode='RGB'))
    plt.show()