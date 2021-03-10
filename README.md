# LAG-Net: Multi-granularity network for person re-identification via local attention system

Code for paper LAG-Net: Multi-granularity network for person re-identification via local attention system

Xun Gong, Zu Yao, Xin Li, Yueqiao Fan, Bin Luo, Jianfeng Fan, and Boji Lao

In IEEE TRANSACTIONS ON MULTIMEDIA. [[pdf]](https://ieeexplore.ieee.org/document/9318538)

```
@ARTICLE{9318538,
  author={X. {Gong} and Z. {Yao} and X. {Li} and Y. {Fan} and B. {Luo} and J. {Fan} and B. {Lao}},
  journal={IEEE Transactions on Multimedia}, 
  title={LAG-Net: Multi-granularity network for person re-identification via local attention system}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3050082}
  }
```


## Introduction
Person re-identification (Re-ID) is a challenging research topic which aims to retrieve the pedestrian images of the same person that captured by non-overlapping cameras. Existing methods either assume the body parts of the same person are well-aligned, or use attention selection mechanisms to constrain the effective region of feature learning. But these methods  concentrate only on coarse feature representation and cannot model complex real scenes effectively. We propose a novel Local Attention Guided Network (LAG-Net) to not only exploit the most salient area among different people, but also extract important local detail through a Local Attention System (LAS). LAS is an attention selection unit that could extract approximate semantic local features of human body parts without extra supervision. To learn discriminative attention feature representation, we explore an attention feature regularization scheme to enhance the relevance of body part features that belong to same personal identity. Considering the effectiveness of feature augmentation in the Re-ID task and the defect of the existing methods, we propose a Batch Attention DropBlock (BA-DropBlock) to further improve DropBlock by combining the attention selection mechanism. Results on mainstream datasets demonstrate the superiority of our model over the state-of-the-art.

## Dependencies
* Python >= 3.5
* PyTorch >= 0.4.0
* TorchVision
* Matplotlib
* Argparse
* Sklearn
* Pillow
* Numpy
* Scipy
* Tqdm

## Train
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ../market1501/ --batchid 8 --batchtest 4 --test_every 50 --epochs 600 --decay_type step_320_380_420 --loss 1*CrossEntropy+2*Triplet+1*L2 --margin 1.2 --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --amsgrad --save lagnet
```
## Result
Results without re-ranking on different datasets.

Market-1501

| mAP | rank@1 | rank@3  | rank@5 | rank@10 |
| --- | --- | --- | --- | --- |
| 89.50 | 95.61 | 97.77 | 98.31 | 99.11 |

DukeMTMC

| mAP | rank@1 | rank@3  | rank@5 | rank@10 |
| --- | --- | --- | --- | --- |
| 81.61 | 90.39 | 94.34 | 96.05 | 97.35 |

CUHK03-np detect
    
| mAP | rank@1 | rank@3  | rank@5 | rank@10 |
| --- | --- | --- | --- | --- |
| 79.1 | 82.4 | 88.2 | 91.6 | 95.1 |

CUHK03-np labeled

| mAP | rank@1 | rank@3  | rank@5 | rank@10 |
| --- | --- | --- | --- | --- |
| 82.16 | 85.14 | 91.14 | 93.79 | 96.57 |

