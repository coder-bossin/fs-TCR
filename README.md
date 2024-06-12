# Towards Better Generalization of Cross-Domain Few-Shot Classification in Tibetan Character with Contrastive Learning and Meta Fine-Tuning
This repo contains codes for few-shot tibetan character recognition and it's solution on cross-domain setting.

## Abstract
 Few-shot classification aims to classify unseen classes (query instances) with few labeled
 samples from each class (support instances). However, current few-shot learning methods assume that the
 training and testing sets obey the same distribution. When there exists a huge domain gap between the training
 and testing sets, they fail to generalize well across domains. In this work, we tackle the cross-domain few
shot learning (CD-FSL) problem in Tibetan characters from two perspectives. In the meta-training phase, we
 seamlessly introduce contrastive learning into the episodic training paradigm and apply a data augmentation
 strategy to seek better feature representations thereby improving the ability to recognize unseen categories.
 In the meta-finetuning phase, we then integrate the above algorithm into transfer learning and propose a
 fine-tuning method that generates episodic synthetic query sets to enhance generalization capability across
 domains. These two stages force the model to overcome the domain shift between training and testing sets.
 Extensive experiments show that our simple approach allows us to establish competitive results on the well
known few-shot learning dataset Omniglot and state-of-the-art results on our Tibetan character datasets.



