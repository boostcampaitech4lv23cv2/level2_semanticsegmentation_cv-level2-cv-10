# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/data/'

# custom classes
classes = ('Backgroud','General trash','Paper','Paper pack','Metal','Glass',
        'Plastic','Styrofoam','Plastic bag','Battery','Clothing')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

albu_train_transforms =[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0,
                rotate_limit=30,
                p=0.5,
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ElasticTransform', p=1.0),
                    dict(type='Perspective', p=1.0),
                    dict(type='PiecewiseAffine', p=1.0),
                ],
                p=0.3),
            dict(
                type='Affine',
              p=0.3  
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RGBShift', r_shift_limit=20, g_shift_limit=20,b_shift_limit=20,always_apply=False,p=1.0),
                    dict(type='ChannelShuffle', p=1.0)
                ],
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
            dict(type='GaussNoise', p=0.3),
            dict(type='CLAHE', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='MedianBlur', blur_limit=5, p=1.0)
                ],
                p=0.3),
        ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
        ),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True,min_size=512),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 1.0, 1.5],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True,min_size=512),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='MultiImageMixDataset',
        dataset = dict(
            type=dataset_type,
            classes=classes,
            palette=palette,
            reduce_zero_label=False,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            pipeline=train_pipeline),
    )
    val=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir='images/test',
        pipeline=test_pipeline))