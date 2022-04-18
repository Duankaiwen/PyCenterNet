_base_ = './pycenternet_r50_fpn_3_levels_giou_fast_mstrain_4x_coco.py'

#learning policy
lr_config = dict(step=[63, 69])
runner = dict(max_epoches=72)
total_epochs = 72  