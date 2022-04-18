### The trained models are temporarily unavailable, but you can train the code using reasonable computational resource.

# [CenterNet++ for Object Detection]()

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Song Bai](http://songbai.site/), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed PyCenterNet is available here. For more technical details, please refer to our [arXiv paper]().**

<div align=center>
<img src=https://github.com/Duankaiwen/PyCenterNet/blob/master/code/demo/PyCenterNet.png width = "1000" height = "300" alt="" align=center />
</div>

## Abstract

  There are two mainstreams for object detection: top-down and bottom-up. The state-of-the-art approaches mostly belong to the first category. In this paper, we demonstrate that the bottom-up approaches are as competitive as the top-down and enjoy higher recall. Our approach, named CenterNet, detects each object as a triplet keypoints (top-left and bottom-right corners and the center keypoint). We firstly group the corners by some designed cues and further confirm the objects by the center keypoints. The corner keypoints equip the approach with the ability to detect objects of various scales and shapes and the center keypoint avoids the confusion brought by a large number of false-positive proposals. Our approach is a kind of anchor-free detector because it does not need to define explicit anchor boxes. We adapt our approach to the backbones with different structures, i.e., the 'hourglass' like networks and the the 'pyramid' like networks, which detect objects on a single-resolution feature map and multi-resolution feature maps, respectively. On the MS-COCO dataset, CenterNet with Res2Net-101 and Swin-Transformer achieves APs of 53.7% and 57.1%, respectively, outperforming all existing bottom-up detectors and achieving state-of-the-art. We also design a real-time CenterNet, which achieves a good trade-off between accuracy and speed with an AP of 43.6% at 30.5 FPS.

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwenduan@outlook.com**

## Bbox AP(%) on COCO test-dev
|Method          |  Backbone       | epoch | MS<sub>train<sub> |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :------------- | :-------:       | :---: | :---------------: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | 
|                |   
| *Anchor-based:*|   
|Libra R-CNN     | X-101-64x4d     | 12  |      N            | 43.0 |     64.0        |       47.0      |      25.3      |      45.6      |      54.6      |
| AB+FSAF*       | X-101-64x4d     | 18  |      Y            | 44.6 |     65.2        |       48.6      |      29.7      |      47.1      |      54.6      |
| FreeAnchor*    | X-101-32x8d     | 24  |      Y            | 47.3 |     66.3        |       51.5      |      30.6      |      50.4      |      59.0      |
| GFLV1*         | X-101-32x8d     | 24  |      Y            | 48.2 |     67.4        |       52.6      |      29.2      |      51.7      |      60.2      |
| ATSS*          | X-101-64x4d-DCN | 24  |      Y            | 50.7 |     68.9        |       56.3      |      33.2      |      52.9      |      62.4      |
| PAA*           | X-101-64x4d-DCN | 24  |      Y            | 51.4 |     69.7        |       57.0      |      34.0      |      53.8      |      64.0      |
| GFLV2*         | R2-101-DCN      | 24  |      Y            | 53.3 |     70.9        |       59.2      |      35.7      |      56.1      |      65.6      |
| YOLOv4-P7*     | CSP-P7          | 450 |      Y            | 56.0 |     73.3        |       61.2      |      38.9      |      60.0      |      68.6      |
|                |   
| *Anchor-free:* |
| ExtremeNet*    | HG-104          | 200 |      Y            | 43.2 |     59.8        |       46.4      |      24.1       |     46.0      |      57.1      | 
| RepPointsV1*   | R-101-DCN       | 24  |      Y            | 46.5 |     67.4        |       50.9      |      30.3       |     49.7      |      57.1      |
| SAPD           | X-101-64x4d-DCN | 24  |      Y            | 47.4 |     67.4        |       51.1      |      28.1       |     50.3      |      61.5      |
| CornerNet*     | HG-104          | 200 |      Y            | 42.1 |     57.8        |       45.3      |      20.8       |     44.8      |      56.7      |
| DETR           | R-101           | 500 |      Y            | 44.9 |     64.7        |       47.7      |      23.7       |     49.5      |      62.3      |
| CenterNet*     | HG-104          | 190 |      Y            | 47.0 |     64.5        |       50.7      |      28.9       |     49.9      |      58.9      |
| CPNDet*        | HG-104          | 100 |      Y            | 49.2 |     67.4        |       53.7      |      31.0       |     51.9      |      62.4      |
| BorderDet*     | X-101-64x4d-DCN | 24  |      Y            | 50.3 |     68.9        |       55.2      |      32.8       |     52.8      |      62.3      |
| FCOS-BiFPN     | X-101-32x8-DCN  | 24  |      Y            | 50.4 |     68.9        |       55.0      |      33.2       |     53.0      |      62.7      |
| RepPointsV2*   | X-101-64x4d-DCN | 24  |      Y            | 52.1 |     70.1        |       57.5      |      34.5       |     54.6      |      63.6      |
|                |
| LSNet          | R-50            | 24  |      Y            | 44.8 |     64.1        |       48.8      |      26.6       |     47.7      |      55.7      |
| LSNet          | X-101-64x4d     | 24  |      Y            | 48.2 |     67.6        |       52.6      |      29.6       |     51.3      |      60.5      |
| LSNet          | X-101-64x4d-DCN | 24  |      Y            | 49.6 |     69.0        |       54.1      |      30.3       |     52.8      |      62.8      |
| LSNet-CPV      | X-101-64x4d-DCN | 24  |      Y            | 50.4 |     69.4        |       54.5      |      31.0       |     53.3      |      64.0      |
| LSNet-CPV      | R2-101-DCN      | 24  |      Y            | 51.1 |     70.3        |       55.2      |      31.2       |     54.3      |      65.0      |
| LSNet-CPV*     | R2-101-DCN      | 24  |      Y            | 53.5 |     71.1        |       59.2      |      35.2       |     56.4      |      65.8      |


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

Generate extreme point annotation from segmentation:
- ```cd code/tools```
- ```python gen_coco_lsvr.py```
- ```cd ..```

## Installation

##### 1. Installing cocoapi 
- ```cd cocoapi/pycocotools```
- ```python setup.py develop```
- ```cd ../..```

##### 2. Installing mmcv 
- ```cd mmcv```
- ```pip install -e.```
- ```cd ..```

##### 3. Installing mmdet 
- ```python setup.py develop```

## Training and Evaluation
Our LSNet is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [with existing dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md) for Training and Evaluation.




