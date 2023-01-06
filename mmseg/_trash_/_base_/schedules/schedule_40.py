# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95)
)
optimizer_config=dict()

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-10,
    )

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(max_keep_ckpts = 3,interval=1)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU', classwise=True)