_base_ = './pycenternet_r50_fpn_giou_mstrain_2x_coco.py'
model = dict(
    pretrained='../checkpoints/pretrained/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth',
    backbone=dict(type='Res2Net',
                  depth=101,
                  scales=4,
                  base_width=26,
                  with_cp=True))
