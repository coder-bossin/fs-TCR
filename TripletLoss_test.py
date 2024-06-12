import os
import random
import torch



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

                    print(loss)
                    total_loss = total_loss+loss

    print(total_loss)

def score_triplet_loss_local(cls_scores, labels_train_1hot):

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    total_loss = 0.0


    cls_scores = cls_scores.view(4, 30, -1)


    for task in range(4):

        supportset = labels_train_1hot[task]

        queryset = cls_scores[task]

        queryset_transform = queryset.view(5, 6, queryset.size(1))

        for classes in range(5):
            cls = [0, 1, 2, 3, 4]
            supportset_singleclass_singleimage = supportset[classes]

            queryset_singleaclass = queryset_transform[classes]

            cls.remove(classes)
            for samplenum in range(6):
                queryset_singleaclass_singleimage = queryset_singleaclass[samplenum]
                negative = random.choice(cls)
                loss = criterion(queryset_singleaclass_singleimage, supportset_singleclass_singleimage, supportset[negative])
                print(loss)
                total_loss = total_loss + loss

    print(total_loss)
    return total_loss



if __name__ == "__main__":
    ftrain_forTripletloss = torch.rand(20,512,11,11)
    ftest_forTripletloss = torch.rand(120,512,11,11)
    patch_triplet_loss(ftrain_forTripletloss, ftest_forTripletloss)
    print("---------------->")
    cls_scores = torch.rand(120,5,1,1)
    labels_train_1hot = torch.rand(4,5,5)
    score_triplet_loss_local(cls_scores, labels_train_1hot)
