import torch
import numpy as np


def patch_loss_global(ytest, pids_test_patch, pids_train):

    criterion = torch.nn.CrossEntropyLoss()

    ytest =  ytest.view( ytest.size()[0],  ytest.size()[1], -1).transpose(1, 2).reshape(-1, ytest.size()[1])
    pids_test_patch =  pids_test_patch.view(-1)
    pids_train_patch = pids_train.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11).view(-1)
    loss = criterion(ytest, torch.cat([pids_train_patch, pids_test_patch.view(-1)]))
    return loss

def patch_loss_local(labels_test_patch, cls_scores):
    '''
    计算infoNCELoss
    '''
    # 实例化交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # torch.Size([4, 30, 11, 11])-->[14520]
    labels_test_patch = labels_test_patch.view(-1)
    # torch.Size([120, 5, 11, 11])-->[120,5,121]-->[120,121,5]-->[14520,5]
    cls_scores = cls_scores.view(cls_scores.size()[0], cls_scores.size()[1], -1).transpose(1,2).reshape(-1, cls_scores.size()[1])
    # 计算模型在每一类（5类别）预测置信度分数与真实标签（0-4）之间的差异
    loss = criterion(cls_scores, labels_test_patch.view(-1))
    return loss

def patch_triplet_loss(ftrain_forTripletloss, ftest_forTripletloss):

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    total_loss = 0.0


    ftrain_forTripletloss = ftrain_forTripletloss.view(4, 5, ftrain_forTripletloss.size(1),
                                                       ftrain_forTripletloss.size(2), ftrain_forTripletloss.size(3))
    ftest_forTripletloss = ftest_forTripletloss.view(4, 30, ftest_forTripletloss.size(1), ftest_forTripletloss.size(2),
                                                     ftest_forTripletloss.size(3))

    for task in range(4):

        supportset = ftrain_forTripletloss[task]

        queryset = ftest_forTripletloss[task]

        queryset_transform = queryset.view(5, 6, queryset.size(1), queryset.size(2), queryset.size(3))

        for classes in range(5):
            cls = [0, 1, 2, 3, 4]

            supportset_singleclass_singleimage = supportset[classes]

            queryset_singleaclass = queryset_transform[classes]

            cls.remove(classes)

            for samplenum in range(6):


                queryset_singleaclass_singleimage = queryset_singleaclass[samplenum]
                for negative in cls:

                    loss = criterion(queryset_singleaclass_singleimage, supportset_singleclass_singleimage,
                                     supportset[negative])

    return total_loss

def generate_matrix():

    xd = np.random.randint(1, 2)
    yd = np.random.randint(1, 2)

    index = list(range(11))

    x0 = np.random.choice(index, size=xd, replace=False)
    y0 = np.random.choice(index, size=yd, replace=False)
    return x0, y0


def random_block(x):

    x0, y0 = generate_matrix()
    mask = torch.zeros([1, 1, 11, 11], requires_grad=False) +1
    for i in x0:
        for j in y0:
                mask[:, :, i, j] = 0
    mask = mask.float()
    x = x * mask.cuda()
    return x

def mid_block(x,mask_size):

    width, height = 84, 84
    left = (width - mask_size) // 2
    lower = (height - mask_size) // 2
    right = left + mask_size
    upper = lower + mask_size
    mask = torch.zeros([20, 1, 84, 84], requires_grad=False) +1
    # double loop：generate mask area(value=0)
    for i in range(left, right):
        for j in range(lower, upper):
                mask[:, :, i, j] = 0
    mask = mask.float()
    x = x * mask.cuda()
    return x

def cross_block(x):

    x0, y0 = 3, 3
    mask = torch.zeros([1, 1, 84, 84], requires_grad=False) +1
    # double loop：generate mask area(value=0)
    for i in x0:
        for j in range(84):
                mask[:, :, 40+i, j] = 0

    for x in y0:
        for y in range(84):
                mask[:, :, y, 40+x] = 0

    mask = mask.float()
    x = x * mask.cuda()
    return x

def rand_bbox(size, lam):

    W = size[2]

    H = size[3]

    cut_rat = np.sqrt(1. - lam)

    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


import os
import os.path as osp
import errno
import json
import shutil

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot
