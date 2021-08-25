# Sampling-Free for Object Detection

*Development, Maintenance @ChenJoya. Please feel free to contact me: chenjoya@foxmail.com*

## Introduction

**To address the foreground-background imbalance, is heuristic sampling necessary in training deep object detectors?** 

Keep clam and try the sampling-free mechanism in this repository. 

**Sampling-free mechanism enables various object detectors (e.g. one-stage, two-stage, anchor-free, multi-stage) to drop sampling heuristics (e.g., undersampling, Focal Loss, objectness), but achieve better bounding-box or instance segmentation accuracy.**

Technical report: https://arxiv.org/abs/1909.04868. This repository is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), including the implementation of RetinaNet/FCOS/Faster/Mask R-CNN. Other detectors will also be released.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Training
See [scripts/train.sh](https://github.com/ChenJoya/sampling-free/blob/master/scripts/train.sh), you can easily train with the sampling-free mechanism.

## Evaluation
See [scripts/eval.sh](https://github.com/ChenJoya/sampling-free/blob/master/scripts/eval.sh), you can easily evaluate your trained model.

### COCO dataset

Model | Config | Box AP (minival) | Mask AP (minival) |
--- |:---:|:---:|:---:|
RetinaNet | retinanet_R_50_FPN_1x | 36.4 | -- |
**RetinaNet + Sampling-Free** | retinanet_R_50_FPN_1x | **36.8** | -- |
FCOS | fcos_R_50_FPN_1x | 37.1 | -- |
**FCOS + Sampling-Free** | fcos_R_50_FPN_1x | **37.6** | -- |
Faster R-CNN | faster_rcnn_R_50_FPN_1x | 36.8 | -- |
**Faster R-CNN + Sampling-Free** | faster_rcnn_R_50_FPN_1x | **38.4** | -- |
Mask R-CNN | mask_rcnn_R_50_FPN_1x | 37.8 | 34.2 |
**Mask R-CNN + Sampling-Free** | mask_rcnn_R_50_FPN_1x | **39.0** | **34.9** |

### PASCAL VOC dataset (07+12 for training)

Model | Config | mAP (07test) |
--- |:---:|:---:|
RetinaNet | retinanet_voc_R_50_FPN_0.2x | 79.3 |
**RetinaNet + Sampling-Free** | retinanet_voc_R_50_FPN_0.2x | **80.1** |
Faster R-CNN | faster_rcnn_voc_R_50_FPN_0.2x | 80.9 |
**Faster R-CNN + Sampling-Free** | faster_rcnn_voc_R_50_FPN_0.2x | **81.5** |

## Other Details
See the original benchmark [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for more details.

## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

```
@article{sampling_free,
author    = {Joya Chen and
             Dong Liu and
             Tong Xu and
             Shiwei Wu and
             Yifei Cheng and
             Enhong Chen},
title     = {Is Heuristic Sampling Necessary in Training Deep Object Detectors?},
journal   = {IEEE Transactions on Image Processing},
year      = {2021},
volume    = {},
number    = {},
pages     = {1-1},
}
```

## License

**sampling-free** is released under the MIT license. See [LICENSE](https://github.com/ChenJoya/sampling-free/blob/master/LICENSE) for additional details.
