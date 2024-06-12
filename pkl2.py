import pickle
import os
import json

import cv2
import pylab
import numpy
from PIL import Image
import numpy as np


def get_image(path):

    i = 0
    j = 0

    images = []
    labels = []


    for dir_item in os.listdir(path):
        full_path = os.path.abspath(os.path.join(path, dir_item))
        if os.path.isdir(full_path):
            get_image(full_path)
        else:
            if dir_item.endswith('.jpg'):

                image = cv2.imread(full_path)
                i+=1
                print('i:',i)
                images.append(image)
                labels.append(dir_item)


    return images, labels


def load_image(path):
    images, labels = get_image(path)
    images = np.array(images)
    print(images.shape)
    labels = np.array([1 if ('wq' in label) else 0 for label in labels])
    print(labels)
    train_dic = {'data': images, 'labels' : labels}
    with open(path+'/data_sample.pickle', 'wb') as f:
        pickle.dump(train_dic, f, 2)
        f.close()

    return images, labels

if __name__ == "__main__":
    # 数据保存路径
    path = r''
    images, labels = load_image(path)
    print(images)
    print(labels)