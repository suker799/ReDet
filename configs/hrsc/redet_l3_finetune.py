# 把上面的 classes = (...) 换成你的 L3 小类全集，work_dir 改成 redet_hrsc_l3；其余保持一致即可。
_base_ = [
    # 以官方 HRSC 配置/或 DOTA 配置为基，路径按你的仓库实际调整
    '../re_det/re50_refpn_1x_hrsc.py'
]

# —— 类别定义（用你的 HRSC L2 小类名，顺序要与标注一致）——
classes = ('ship_type_A', 'ship_type_B', 'ship_type_C', 'ship_type_D')

# —— 数据集（DOTA/HRSC风）——
dataset_type = 'DOTADataset'  # HRSC 常用 DOTA 风 OBB 读取
data_root = 'data/hrsc/'      # 下面3个目录自己准备好
img_dir = dict(train='train/images', val='val/images', test='test/images')
ann_dir = dict(train='train/labels', val='val/labels', test='test/labels')  # DOTA 格式 txt/OBB

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 几何增强（关闭颜色增强，光学/灰度更稳）
    dict(type='RandomRotate', angle_range=15, auto_bound=False, pad_val=0),
    dict(type='Resize', img_scale=[(768,768),(896,896),(1024,1024)], multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical']),
    dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_bboxes','gt_bboxes_ignore','gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1024,1024),
         flip=False,
         transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RotatedRandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=2, workers_per_gpu=2,
    train=dict(type=dataset_type, ann_file=data_root+ann_dir['train'], img_prefix=data_root+img_dir['train'], classes=classes, pipeline=train_pipeline),
    val  =dict(type=dataset_type, ann_file=data_root+ann_dir['val'],   img_prefix=data_root+img_dir['val'],   classes=classes, pipeline=test_pipeline),
    test =dict(type=dataset_type, ann_file=data_root+ann_dir['test'],  img_prefix=data_root+img_dir['test'],  classes=classes, pipeline=test_pipeline),
    )

# —— 模型头部多类化 + 损失偏召回 —— 
model = dict(
    bbox_head=dict(
        num_classes=len(classes),
        loss_cls=dict(  # Focal 更抗不平衡，召回↑
            type='FocalLoss', use_sigmoid=True, gamma=1.5, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    )
)

# —— 训练策略：先冻 backbone → 再全网 —— 
# 第1阶段（只训头）
freeze_backbone_epochs = 8
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-5)
total_epochs = 36

# —— 针对召回的 assigner/sampler 小调整 —— 
train_cfg = dict(
    rpn=dict(assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.4)),
    rcnn=dict(
        assigner=dict(pos_iou_thr=0.5, neg_iou_thr=0.4),
        sampler=dict(num=512, pos_fraction=0.25)))
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,  # 推理期我们会另行网格扫描
        nms=dict(iou_thr=0.5),
        max_per_img=2000)
)

# —— 载入单类HRSC权重，仅作骨干初始化（检测头会重置）——
load_from = 'checkpoints/redet_hrsc.pth'
work_dir = './work_dirs/redet_hrsc_l3'
