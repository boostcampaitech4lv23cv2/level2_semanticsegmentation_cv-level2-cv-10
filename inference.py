import os
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataloaders.DataLoader import CustomDataLoader

import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2

import yaml
from easydict import EasyDict

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import multiprocessing as mp
import torch.nn.functional as F

import ttach as tta

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])

def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(encoder, encoder_weights, decoder, device):
    model_module = getattr(smp, decoder)
    model = model_module(
        encoder_name=encoder,
        encoder_weights=encoder_weights,     
        in_channels=3,               
        classes=11,                
        encoder_output_stride=32 # Mit model     
    )
    # model = getattr(import_module("model"), "getModel")()
    model_path = './saved/PAN+mit_5_fix.pt'
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    return model


def test(dataset_path, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_path = dataset_path + '/test.json'
    test_transform = A.Compose([
                            ToTensorV2()
                           ])
    test_dataset = CustomDataLoader(data_dir=test_path, dataset_path=dataset_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction!!')

    model = load_model(args.encoder, args.encoder_weights, args.decoder, device).to(device)

    if args.tta:
        tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90])   
        ]
        )
        tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode = 'mean')
        tta_model.eval()
    else :
        model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            imgs = torch.stack(imgs)
            if args.tta:
                logits = tta_model(imgs.to(device))
            else:
                logits = model(imgs.to(device))
            logits = F.interpolate(
                    logits, size=imgs.shape[2:], mode="bilinear", align_corners=True
                )
            probs = F.softmax(logits, dim=1)
            probs = probs.data.cpu().numpy()
            
            pool = mp.Pool(mp.cpu_count())
            imgs = imgs.data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            probs = pool.map(dense_crf_wrapper, zip(imgs, probs))
            pool.close()
            probs = torch.from_numpy(np.array(probs)).float().cuda()
            
            outs = probs
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # # inference (512 x 512)
            # if args.tta:
            #     outs = tta_model(torch.stack(imgs).to(device))
            # else:
            #     outs = model(torch.stack(imgs).to(device))
            # oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


dataset_path = '/opt/ml/input/data'
CONFIG_FILE_NAME = "./config/config.yaml"
with open(CONFIG_FILE_NAME, "r") as yml_config_file:
    args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
    args = EasyDict(args["test"])

os.makedirs(args.output_dir, exist_ok=True)

# sample_submisson.csv 열기
submission = pd.read_csv(os.path.join(args.output_dir, 'sample_submission.csv'), index_col=None)

# test set에 대한 prediction
file_names, preds = test(dataset_path, args)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(args.output_dir, 'pan_mit_b5_crf_ttaV2.csv'), index=False)