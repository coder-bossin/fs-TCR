from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io
import random

import torch
from torch.utils.data import Dataset
from copy import deepcopy


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            #img = Image.open(img_path).convert('RGB')
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_finetune(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
    """

    def __init__(self,
                 dataset,  # dataset of [(img_path, cats), ...].
                 labels2inds,  # labels of index {(cats: index1, index2, ...)}.
                 labelIds,  # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5,  # number of novel categories.
                 nExemplars=1,  # number of training examples per novel category.
                 nTestNovel=5 * 6,  # number of test examples for all the novel categories.
                 epoch_size=2000,  # number of tasks per epoch
                 transform=None,
                 load=True,
                 pseudo_q_genrator=None,
                 **kwargs
                 ):

        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load
        self.pseudo_q_genrator = pseudo_q_genrator


    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """
        Knovel = random.sample(self.labelIds, self.nKnovel)
        nKnovel = len(Knovel)
        assert ((self.nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)


        Exemplars = []
        for Knovel_idx in range(len(Knovel)):

            ids = self.nExemplars

            imgs_emeplars_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids)

            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars_ids]
        assert (len(Exemplars) == nKnovel * self.nExemplars)

        return Exemplars

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
        """

        images_support = []
        labels_support = []
        images_support_cls = []
        images_query = []
        labels_query = []
        images_query_cls = []

        for (img_idx, label) in examples:

            img, cls = self.dataset[img_idx]

            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            # yes
            if self.transform is not None:
                img = self.transform(img)


            pseudo_query_set_imgs, pseudo_query_labels, pseudo_query_cls = self.pseudo_q_genrator.generate(img, label, cls)


            images_support.append(img)
            labels_support.append(label)
            images_support_cls.append(cls)

            images_query.extend(pseudo_query_set_imgs)

            labels_query.extend(pseudo_query_labels)
            images_query_cls.extend(pseudo_query_cls)

        images_support = torch.stack(images_support, dim=0)
        labels_support = torch.LongTensor(labels_support)
        images_support_cls = torch.LongTensor(images_support_cls)

        images_query = torch.stack(images_query, dim=0)
        labels_query = torch.LongTensor(labels_query)
        images_query_cls = torch.LongTensor(images_query_cls)

        return images_support, labels_support, images_query, labels_query, images_support_cls, images_query_cls

    def __getitem__(self, index):

        Exemplars = self._sample_episode()
        Xt, Yt, Xe, Ye, Ytc, Yec = self._creatExamplesTensorData(Exemplars)
        return Xt, Yt, Xe, Ye, Ytc, Yec,

