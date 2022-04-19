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
| PyCenterNet    | R-50            | 46.4 |     63.7        |       50.3      |      27.1      |      48.9      |      58.8      |       36.2     |       60.0      |     64.2       |      41.1       | 68.5 |     81.9        |



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




