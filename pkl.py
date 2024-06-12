import pickle
import os
import json
import cv2
import pylab
import numpy
from PIL import Image
import numpy as np
import joblib



def trans_data(imageFile, img_num, label_num):

    data = np.empty((img_num, 84, 84))
    data_label = np.empty(img_num)

    classes = []
    for i in range(10):
        classes.append(i)
    classes = list(map(str, classes))

    '''
    labels = []
    for i in range(20):
        labels.append(i)
    labels = list(map(str, labels))
    '''

    i = 0
    j = 0

    for num in range(label_num):
        classfile = imageFile + '\\' + classes[j]
        classfile = classfile.replace('\\', '/')

        for filename in os.listdir(classfile):
            if (filename != 'Thumbs.db'):
                basedir = classfile + '/'
                img = cv2.imread(basedir + filename, 0)
                img = cv2.resize(img, (84, 84))
                img_ndarray = numpy.asarray(img, dtype='uint8')
                data[i] = img_ndarray
                i = i + 1
        j = j + 1
    data_label = np.array(data_label)
    return data, data_label
def trans_data_omniglot(imageFile, img_num, label_num):

    data = np.empty((img_num, 84, 84))
    data_label = np.empty(img_num)
    classes = []
    for i in range(984):
        classes.append(i)
    classes = list(map(str, classes))

    i = 0
    j = 0


    for language in os.listdir(imageFile):

        languagefile = imageFile + '\\' + language
        languagefile = languagefile.replace('\\', '/')

        for character in os.listdir(languagefile):

            characterfile = languagefile+ '\\' + character
            characterfile = characterfile.replace('\\', '/')

            for characterimage in os.listdir(characterfile):
                if (characterimage != 'Thumbs.db'):
                    basedir = characterfile + '/'
                    img = cv2.imread(basedir + characterimage, 0)
                    img = cv2.resize(img, (84, 84))
                    img_ndarray = numpy.asarray(img, dtype='uint8')
                    data[i] = img_ndarray
                    data_label[i] = int(classes[j])
                    i = i + 1
            j = j + 1
    data_label = np.array(data_label)
    return data, data_label




def pic_pkl(data, data_label):


    data_dic = {'data': data,
                'labels': data_label}
    with open('', 'wb') as f:

        pickle.dump(data_dic, f, 2)

        f.close()




def count_subdirectories(folder_path):

    count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            for item in os.listdir(item_path):
                sub_item = os.path.join(item_path, item)
                if not any(os.path.isdir(item) for item in os.listdir(sub_item)):
                    count += 1
    return count

if __name__ == "__main__":
    imageFile = r''
    img_num = (sum([len(x) for _, _, x in os.walk(imageFile)]))
    label_num = len(os.listdir(imageFile))

    data, data_label = trans_data(imageFile, img_num, label_num)
    pic_pkl(data, data_label)




