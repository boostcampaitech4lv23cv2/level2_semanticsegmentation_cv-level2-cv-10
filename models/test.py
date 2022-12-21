import segmentation_models_pytorch as smp

# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=4,                 # define number of output labels
# )

model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b8",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    encoder_depth=5,
    decoder_channels=[256, 128, 64, 32, 16],
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=11,                      # model output channels (number of classes in your dataset)
    activation=None,
    # aux_params=aux_params,
)


#https://www.kaggle.com/code/alepru/kddm2-unet-train-inf
#https://smp.readthedocs.io/en/latest/models.html#unet