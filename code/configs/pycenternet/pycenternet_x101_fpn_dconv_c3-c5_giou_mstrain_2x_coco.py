_base_ = './pycenternet_r50_fpn_giou_mstrain_2x_coco.py'
model = dict(
    pretrained='../checkpoints/pretrained/resnext101_64x4d-ee2c6f71.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'))


########### multi-scale testing, we follow ATSS, https://github.com/sfzhang15/ATSS #############

# test_cfg = dict(method = 'vote',
#                 scale_ranges = [[96, 10000], [96, 10000], [64, 10000], [64, 10000],
#                                 [64, 10000], [0, 10000], [0, 10000], [0, 256], [0, 256],
#                                 [0, 192], [0, 192], [0, 96]])

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(3000, 400), (3000, 500), (3000, 600), (3000, 640), (3000, 700), (3000, 900),
#                    (3000, 1000), (3000, 1100), (3000, 1200), (3000, 1300), (3000, 1400), (3000, 1800)],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(test=dict(pipeline=test_pipeline))