import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from resnet import resnet12
import transforms as T
import numpy as np

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from resnet import resnet12
import transforms as T


def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


def draw_features2(width, height, x, savefolder):
    tic = time.time()

    for i in range(width * height):
        fig, ax = plt.subplots(figsize=(4, 4))

        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]
        ax.imshow(img)

        savepath = f"{savefolder}/feature_{i + 1}.jpg"
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print("{}/{}".format(i + 1, width * height))

    print("Total time: {}".format(time.time() - tic))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet12()

model = model.to(device)



transform_train = T.Compose([
    transforms.Resize((84, 84)),

    T.Grayscale(1),


    T.ToTensor(),
    # omniglot
    # T.Normalize(mean=[0.922], std=[0.262]),
    # Tibetan_fewshot
    T.Normalize(mean=[0.437], std=[0.483]),

])

# 加载图像
img_path = r''
img = Image.open(img_path)
img_tensor = transform_train(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(device)
print(img_tensor.size())

with torch.no_grad():
    output = model(img_tensor)

print(output.size())
output = output.cpu().numpy()
draw_features2(8,8,output,r'')