_base_ = './pycenternet_r50_fpn_giou_fast_mstrain_4x_coco.py'

model=dict(
    neck=dict(
        num_outs=3),
    bbox_head=dict(
        point_strides=[8, 16, 32]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(900, 256), (900, 608)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadRPDV2Annotations'),
    dict(type='RPDV2FormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights']),
]

data = dict(train=dict(pipeline=train_pipeline))

                 