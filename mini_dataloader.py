from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

import transforms as T
import mini_dataset
import sampler.mini_sampler_test as sample_test
import sampler.mini_sampler_train as sample_train
import sampler.mini_sampler_finetune as sample_finetune

class DataManager(object):
    """
    Few shot data manager
    """
    def __init__(self, args, use_gpu, pseudo_q_generator):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu
        self.pseudo_q_generator = pseudo_q_generator

        print("Initializing dataset {}".format(args.dataset))
        dataset = mini_dataset.miniImageNet_load()
        transform_train = T.Compose([

                T.Grayscale(1),


                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229]),

            ])
        # 测试数据变换形式
        transform_test = T.Compose([

                T.Grayscale(1),

                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229]),
            ])

        pin_memory = True if use_gpu else False

        self.trainloader = DataLoader(
                sample_train.FewShotDataset_train(name='train_loader',
                    dataset=dataset.train,# data_pair
                    labels2inds=dataset.train_labels2inds,
                    labelIds=dataset.train_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.train_nTestNovel,
                    epoch_size=args.train_epoch_size,
                    transform=transform_train,
                    load=args.load,
                ),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=True,
            )
        self.testloader = DataLoader(
                sample_test.FewShotDataset_test(name='test_loader',
                    dataset=dataset.test,
                    labels2inds=dataset.test_labels2inds,
                    labelIds=dataset.test_labelIds,
                    nKnovel=args.nKnovel,
                    nExemplars=args.nExemplars,
                    nTestNovel=args.nTestNovel,
                    epoch_size=args.epoch_size,
                    transform=transform_test,
                    load=args.load,
                ),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                pin_memory=pin_memory, drop_last=False,
        )

        self.finetuneloader = DataLoader(
            sample_finetune.FewShotDataset_finetune(name='finetune_loader',
                                            dataset=dataset.test,
                                            labels2inds=dataset.test_labels2inds,
                                            labelIds=dataset.test_labelIds,
                                            nKnovel=args.nKnovel,
                                            nExemplars=args.nExemplars,
                                            nTestNovel=args.nTestNovel,
                                            epoch_size=args.epoch_size,
                                            transform=transform_test,
                                            load=args.load,
                                            pseudo_q_generator=pseudo_q_generator
                                            ),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )
    def return_dataloaders(self):
        return self.trainloader, self.testloader, self.finetuneloader
