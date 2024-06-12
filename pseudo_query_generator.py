import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import random
import cv2
import os


def random_aug(x):
    # gamma correction
    if random.random() <= 0.3:
        gamma = random.uniform(1.0, 1.5)
        x = gamma_correction(x, gamma)
    # random erasing with mean value
    mean_v = tuple(x.view(x.size(0), -1).mean(-1))
    re = transforms.RandomErasing(p=0.5, value=mean_v)
    x = re(x)

    if random.random() <= 0.5:
        # degree = [90, 180, 270]
        degree = [-45, -30, -15, -5, 5, 15, 30, 45]
        d = random.choice(degree)
        # x = torch.rot90(x, d // 90, [1, 2])
        x = transforms.functional.rotate(x, d)

    return x


class PseudoQeuryGenerator(object):
    def __init__(self, n_way, n_support, n_EvalExamplesPerClass):
        super(PseudoQeuryGenerator, self).__init__()

        self.n_way = n_way
        self.n_support = n_support
        self.n_EvalExamplesPerClass = n_EvalExamplesPerClass

    def generate(self, support_set_img, label, cls):

        times = self.n_EvalExamplesPerClass

        pseudo_query_imgs = []
        pseudo_query_labels = []
        pseudo_query_cls = []


        for i in range(times):
            support_set_img = random_aug(support_set_img)
            pseudo_query_imgs.append(support_set_img)
            pseudo_query_labels.append(label)
            pseudo_query_cls.append(cls)

        return pseudo_query_imgs, pseudo_query_labels, pseudo_query_cls

def gamma_correction(x, gamma):
    minv = torch.min(x)
    x = x - minv
    maxv = torch.max(x)

    x = x / maxv
    x = x ** gamma
    x = x * maxv
    x = x - minv
    return x


image_path = r''
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_tensor = transforms.ToTensor()(image)

save_path = r''
for i in range(10):
    augmented_image_tensor = random_aug(image_tensor)

    file_name = f"11_aug{i+1}.jpg"

    augmented_image = augmented_image_tensor.permute(1, 2, 0).numpy()
    cv2.imshow("Augmented Image", augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_file_path = os.path.join(save_path, file_name)

    cv2.imwrite(save_file_path, augmented_image * 255)





