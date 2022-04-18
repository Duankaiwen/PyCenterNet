### The trained models are temporarily unavailable, but you can train the code using reasonable computational resource.

# [CenterNet++ for Object Detection]()

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Song Bai](http://songbai.site/), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed PyCenterNet is available here. For more technical details, please refer to our [arXiv paper]().**

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/lsvr.png width = "600" height = "250" alt="" align=center />
</div>

## Abstract

  Object detection, instance segmentation, and pose estimation are popular visual recognition tasks which require localizing the object by internal or boundary landmarks. This paper summarizes these tasks as location-sensitive visual recognition and proposes a unified solution named location-sensitive network (LSNet). Based on a deep neural network as the backbone, LSNet predicts an anchor point and a set of landmarks which together define the shape of the target object. The key to optimizing the LSNet lies in the ability of fitting various scales, for which we design a novel loss function named cross-IOU loss that computes the cross-IOU of each anchor-landmark pair to approximate the global IOU between the prediction and groundtruth. The flexibly located and accurately predicted landmarks also enable LSNet to incorporate richer contextual information for visual recognition. Evaluated on the MSCOCO dataset, LSNet set the new state-of-the-art accuracy for anchor-free object detection (a 53.5% box AP) and instance segmentation (a 40.2% mask AP), and shows promising performance in detecting multi-scale human poses. 

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

*A comparison between LSNet and the sate-of-the-art methods in object detection on the MS-COCO test-dev set. LSNet surpasses all competitors in the anchor-free group. The abbreviations are: ‘R’ – ResNet, ‘X’ – ResNeXt, ‘HG’ – Hourglass network, ‘R2’ – Res2Net, ‘CPV’ – corner point verification, ‘MStrain’ – multi-scale training, * – multi-scale testing.*
             
## Segm AP(%) on COCO test-dev
|Method            |  Backbone       | epoch |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :-------------   | :-------:       | :---: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | 
|                  |   
| *Pixel-based:*   |   
| YOLACT           | R-101           | 48    | 31.2 |      50.6       |     32.8        |       12.1      |      33.3      |      47.1     |
| TensorMask       | R-101           | 72    | 37.1 |      59.3       |     39.4        |       17.1      |      39.1      |      51.6     |
| Mask R-CNN       | X-101-32x4d     | 12    | 37.1 |      60.0       |     39.4        |       16.9      |      39.9      |      53.5     |
| HTC              | X-101-64x4d     | 20    | 41.2 |      63.9       |     44.7        |       22.8      |      43.9      |      54.6     |
| DetectoRS*       | X-101-64x4d     | 40    | 48.5 |      72.0       |     53.3        |       31.6      |      50.9      |      61.5     |
|                  |   
| *Contour-based:* |
| ExtremeNet       | HG-104          | 100   | 18.9 |      44.5       |      13.7       |        10.4     |      20.4      |      28.3     |
| DeepSnake        | DLA-34          | 120   | 30.3 |       -         |       -         |         -       |       -        |        -      |
| PolarMask        | X-101-64x4d-DCN | 24    | 36.2 |      59.4       |      37.7       |        17.8     |      37.7      |      51.5     |
|                  |
| LSNet            | X-101-64x4d-DCN | 30    | 37.6 |      64.0       |      38.3       |        22.1     |      39.9      |      49.1     |
| LSNet            | R2-101-DCN      | 30    | 38.0 |      64.6       |      39.0       |        22.4     |      40.6      |      49.2     |
| LSNet*           | X-101-64x4d-DCN | 30    | 39.7 |      65.5       |      41.3       |        25.5     |      41.3      |      50.4     |
| LSNet*           | R2-101-DCN      | 30    | 40.2 |      66.2       |      42.1       |        25.8     |      42.2      |      51.0     |

*Comparison of LSNet to the sate-of-the-art methods in instance segmentation task on the COCO test-dev set. Our LSNet achieves the state-of-the-art accuracy for contour-based instance segmentation. ‘R’ - ResNet, ‘X’ - ResNeXt, ‘HG’ - Hourglass, ‘R2’ - Res2Net, * -  multi-scale testing.*

## Keypoints AP(%) on COCO test-dev
|Method               |  Backbone       | epoch |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :-------------      | :-------:       | :---: | :--: | :-------------: | :-------------: | :------------: | :------------: | 
|                     |   
| *Heatmap-based:*    |   
| CenterNet-jd        |  DLA-34         | 320   | 57.9 |      84.7       |      63.1       |       52.5     |      67.4      |
| OpenPose            |  VGG-19         | -     | 61.8 |      84.9       |      67.5       |       58.0     |      70.4      |
| Pose-AE             |  HG             | 300   | 62.8 |      84.6       |      69.2       |       57.5     |      70.6      |
| CenterNet-jd        |  HG104          | 150   | 63.0 |      86.8       |      69.6       |       58.9     |      70.4      |
| Mask R-CNN          |  R-50           | 28    | 63.1 |      87.3       |      68.7       |       57.8     |      71.4      |
| PersonLab           |  R-152          | >1000 | 66.5 |      85.5       |      71.3       |       62.3     |      70.0      |
| HRNet               |  HRNet-W32      | 210   | 74.9 |      92.5       |      82.8       |       71.3     |      80.9      |
|                     |   
| *Regression-based:* |
| CenterNet-reg       |  DLA-34         | 320   | 51.7 |      81.4       |       55.2      |       44.6     |      63.0      |
| CenterNet-reg       |  HG-104         | 150   | 55.0 |      83.5       |       59.7      |       49.4     |      64.0      |
|                     |
| LSNet w/ obj-box    |  X-101-64x4d-DCN| 60    | 55.7 |      81.3       |       61.0      |       52.9     |      60.5      |
| LSNet w/ kps-box    |  X-101-64x4d-DCN| 20    | 59.0 |      83.6       |       65.2      |       53.3     |      67.9      |

*Comparison of LSNet to the sate-of-the-art methods in pose estimation task on the COCO test-dev set. LSNet
predict the keypoints by regression. ‘obj-box’ and ‘kps-box’ denote the object bounding boxes and the keypoint-boxes,
respectively. For LSNet w/ kps-box, we fine-tune the model from the LSNet w/ kps-box for another 20 epochs.*

## Visualization

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/dect-segm-pose.png width = "1000" height = "250" alt="" align=center />
  
*Some location-sensitive visual recognition results on the MS-COCO validation set.*
</div>


<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/pose1.png width = "400" height = "350" alt="" align=center />
  
*We compared with the CenterNet to show that our LSNet w/ ‘obj-box’ tends to predict more human pose of small scales, which are not annotated on the dataset. Only pose results with scores higher than 0:3 are shown for both methods.*
</div>

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/pose2.png width = "500" height = "800" alt="" align=center />
  
*Left: LSNet uses the object bounding boxes to assign training samples. Right: LSNet uses the keypoint-boxes to
assign training samples. Although LSNet with keypoint-boxes enjoys higher AP score, its ability of perceiving multi-scale
human instances is weakened.*
</div>

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




