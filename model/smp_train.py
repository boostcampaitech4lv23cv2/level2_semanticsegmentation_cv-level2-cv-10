import segmentation_models_pytorch as smp
import os
import torch
import matplotlib.pyplot as Plastic
from pprint import pprint
from torch.utils.data import DataLoader

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=4,                 # define number of output labels
)

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    encoder_depth=4,
    decoder_channels=[256, 128, 64, 32],
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
    activation=None,
    aux_params=aux_params
)

i = torch.rand(2, 1, 64, 64)
print(model(i))

#https://www.kaggle.com/code/alepru/kddm2-unet-train-inf
#https://smp.readthedocs.io/en/latest/models.html#unet