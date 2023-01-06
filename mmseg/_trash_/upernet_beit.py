_base_ = [
    './_base_/datasets/custom.py',
    './_base_/models/upernet_beit.py',
    './_base_/schedules/schedule_160k.py',
    './_base_/default_runtime.py'
]

# checkpoint_file = '/opt/ml/input/mmseg/_pretrained_/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'  # noqa

model = dict(
    pretrained='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='BEiT',
        img_size=512,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=11, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05)

data = dict(samples_per_gpu=3)
