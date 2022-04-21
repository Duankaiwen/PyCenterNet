## For the Hourglass based CenterNet, please refer to [HgCenterNet](https://github.com/Duankaiwen/CenterNet)
### The trained models are temporarily unavailable, but you can train the code using reasonable computational resource.

# [CenterNet++ for Object Detection](https://arxiv.org/abs/2204.08394)

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Song Bai](http://songbai.site/), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed PyCenterNet is available here. For more technical details, please refer to our [arXiv paper](https://arxiv.org/abs/2204.08394).**

<div align=center>
<img src=https://github.com/Duankaiwen/PyCenterNet/blob/master/code/demo/PyCenterNet.png width = "1000" height = "300" alt="" align=center />
</div>

## Abstract

  There are two mainstreams for object detection: top-down and bottom-up. The state-of-the-art approaches mostly belong to the first category. In this paper, we demonstrate that the bottom-up approaches are as competitive as the top-down and enjoy higher recall. Our approach, named CenterNet, detects each object as a triplet keypoints (top-left and bottom-right corners and the center keypoint). We firstly group the corners by some designed cues and further confirm the objects by the center keypoints. The corner keypoints equip the approach with the ability to detect objects of various scales and shapes and the center keypoint avoids the confusion brought by a large number of false-positive proposals. Our approach is a kind of anchor-free detector because it does not need to define explicit anchor boxes. We adapt our approach to the backbones with different structures, i.e., the 'hourglass' like networks and the the 'pyramid' like networks, which detect objects on a single-resolution feature map and multi-resolution feature maps, respectively. On the MS-COCO dataset, CenterNet with Res2Net-101 and Swin-Transformer achieves APs of 53.7% and 57.1%, respectively, outperforming all existing bottom-up detectors and achieving state-of-the-art. We also design a real-time CenterNet, which achieves a good trade-off between accuracy and speed with an AP of 43.6% at 30.5 FPS.

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwenduan@outlook.com**

## Bbox AP(%) on COCO test-dev
|Method          |  Backbone       | AP  | AP<sub>50</sub>  | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | AR<sub>1</sub> | AR<sub>10</sub> | AR<sub>100</sub> | AR<sub>S</sub>  | AR<sub>M</sub> | AR<sub>L</sub> |
| :------------- | :-------:       | :--:| :-------------:  | :-------------: | :------------: | :------------: | :------------: | :------------: | :------------:  | :------------:   | :------------:  | :------------: | :------------: |
| PyCenterNet    | [R-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)                                                    | 46.4 |     63.7        |       50.3      |      27.1      |      48.9      |      58.8      |       36.2     |       60.0      |     64.2       |      41.1        | 68.5 |     81.9        |
| PyCenterNet    | [R-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)                                                  | 47.7 |     65.1        |       51.9      |      27.8      |      50.5      |      60.6      |       37.1     |       61.1   |         65.4       |      41.6       | 70.0 |     83.4        |
| PyCenterNet    | [R-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)-DCN                                              | 49.8 |     67.3        |       54.1      |      29.1      |      52.6      |      64.2      |       37.8     |       62.0   |         66.3       |      43.6       | 70.8 |     84.0        |
| PyCenterNet    | [X-101](https://download.openmmlab.com/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth)                           | 49.3 |     67.0        |       53.7      |      30.1      |      52.2      |      62.1      |       37.5     |       61.8   |         66.0       |      43.9       | 70.2 |     83.2        |
| PyCenterNet    | [X-101](https://download.openmmlab.com/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth)-DCN                       | 50.8 |     68.6        |       55.4      |      30.7      |      53.4      |      65.3      |       38.2     |       62.7   |         66.9       |      44.9       | 71.0 |     84.6        |
| PyCenterNet    | [R2-101](https://download.openmmlab.com/pretrain/third_party/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth)             | 50.2 |     67.9        |       54.7      |      30.5      |      53.4      |      63.2      |       38.1     |       62.7   |         67.0       |      44.8       | 71.6 |     84.0        |
| PyCenterNet    | [R2-101](https://download.openmmlab.com/pretrain/third_party/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth)-DCN         | 51.5 |     69.2        |       56.2      |      31.0      |      54.4      |      65.7      |       38.5     |       63.1   |         67.5       |      45.6       | 71.7 |     84.6        |
| PyCenterNet(MS)| [R2-101](https://download.openmmlab.com/pretrain/third_party/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth)-DCN         | 53.7 |     70.9        |       59.7      |      35.1      |      56.0      |      66.7      |       39.8     |       66.6   |         71.8       |      54.3       | 74.5 |     86.2        |
| PyCenterNet    | [Swin-L](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth) | 53.2 |     71.4        |       57.4      |      33.2      |      56.2      |      68.7      |       39.2     |       61.6   |         64.0       |      43.2       | 67.7 |     80.7        |
| PyCenterNet(MS)| [Swin-L](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth) | 57.1 |     73.7        |       62.4      |      38.7      |      59.2      |      71.3      |       40.9     |       67.4   |         72.2       |      54.8       | 75.1 |     86.8        |

*'MS'– multi-scale testing*


## Performance of the Real-time CenterNet on COCO test-dev
|Method             |  Backbone       | FPS | AP<sub>val</sub>  | AP<sub>test</sub> |
| :-------------    | :-------:       | :--:| :-------------:   | :-------------:   | 
| YOLOv3            | Darknet-53      | 26  |     -             |       33.0        |  
| FCOS-RT           | R-50            | 38  |     40.2          |       40.2        | 
| Objects as Points | DLA-34          | 52  |     37.4          |       37.3        |  
| CPNDet            | DLA-34          | 26.2|     41.6          |       41.8        |
|                   |
| CenterNet-RT      | R-50            | 30.5|     43.2          |       43.6        |


## Preparation
The master branch works with PyTorch 1.5.0

The dataset directory should be like this:
```plain
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── images
            ├── train2017
            ├── val2017
            ├── test2017
```
## Installation
```cd code```
##### 1. Installing cocoapi 
- ```cd mmpycocotools```
- ```python setup.py develop```
- ```cd ..```

##### 2. Installing mmcv 
- ```cd mmcv```
- ```MMCV_WITH_OPS=1 pip install -e .```
- ```cd ..```

##### 3. Installing mmdet 
- ```python setup.py develop```

## Training and Evaluation
Our CenterNet is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [with existing dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md) for Training and Evaluation.




