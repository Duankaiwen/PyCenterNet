_base_ = './pycenternet_r50_fpn_giou_1x_coco.py'

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    bbox_head=dict(
        type='PyCenterNetFastHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=1,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1),
        _delete_=True))

lr_config = dict(step=[30, 34])
total_epochs = 36       
# multi-scale training
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(736, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(train= dict(pipeline=train_pipeline),
            val  = dict(pipeline=test_pipeline),
            test = dict(pipeline=test_pipeline))

                 