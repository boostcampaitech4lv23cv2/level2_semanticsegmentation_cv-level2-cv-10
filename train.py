from importlib import import_module
from pathlib import Path

import os
import random
import time
import json
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders.DataLoader import CustomDataLoader
from loss import create_criterion
from scheduler import create_scheduler

import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import MetricCollection

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
import yaml
from easydict import EasyDict

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model(model, saved_dir, file_name='best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def collate_fn(batch):
    return tuple(zip(*batch))


def save_table(table_name, model, val_loader, device):
  table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types = True)

  for step, (im, mask, _) in tqdm(enumerate(val_loader), total = len(val_loader)):

    im = torch.stack(im)       
    mask = torch.stack(mask).long()

    im, mask, = im.to(device), mask.to(device)

    _mask = model(im)
    _, _mask = torch.max(_mask, dim=1)

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(im[0].permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_image.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_mask.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(_mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("predicted_mask.jpg")
    plt.close()

    table.add_data(
        wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
    )


  wandb.log({table_name: table})


def train(args):
    print(f'Start training...')

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentations
    train_transform = A.Compose([
                            ToTensorV2()
                            ])

    val_transform = A.Compose([
                          ToTensorV2()
                          ])
    # -- data_set
    train_path = dataset_path + args.train_path
    val_path = dataset_path + args.val_path

    train_dataset = CustomDataLoader(data_dir=train_path, dataset_path=dataset_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, dataset_path=dataset_path, mode='val', transform=val_transform)

    # -- datalodaer
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.valid_batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
                                         
    # -- model
    model_module = getattr(smp, args.decoder)
    model = model_module(
        encoder_name=args.encoder, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                     # model output channels (number of classes in your dataset)
    )
    # device 할당
    model = model.to(device)   

    # -- loss & metric
    criterion = create_criterion(args.criterion)

    # -- optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.lr * 100
    )

    # # -- scheduler
    if args.scheduler:
        scheduler = create_scheduler(optimizer, args.scheduler, args.epochs, args.lr)

    # -- train
    n_class = 11
    best_loss, best_mIoU = np.inf, 0
    val_every = 1
    
    # Grad accumulation
    NUM_ACCUM = args.grad_accum
    optimizer.zero_grad()
    
    # Early Stopping
    PATIENCE = args.patience
    counter = 0

    # average = macro가 기본 옵션입니다
    hist = MulticlassJaccardIndex(num_classes=n_class).cuda()
    for epoch in range(args.epochs):
        model.train()

        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                        
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            loss.backward()

            if step % NUM_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # batch에 대한 mIoU 계산, baseline code는 누적을 계산합니다
            mIoU = hist(outputs, masks).item()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_interval == 0:
                current_lr = get_lr(optimizer)
                print(f'Epoch [{epoch+1}/{args.epochs}] || Step [{step+1}/{len(train_loader)}] || Loss: {round(loss.item(),4)} || mIoU: {round(mIoU,4)}')
                # wandb
                wandb.log(
                    {'Tr Loss': loss.item(), 'Tr mIoU': mIoU, 'lr': current_lr}
                )
        
        hist.reset()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU, IoU_by_class = validation(model, val_loader, device, criterion, epoch, args)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = val_mIoU
                save_model(model, saved_dir)
                counter = 0
            else:
                counter += 1

            # wandb
            wandb.log(
                {'Val Loss': avrg_loss, 'Val mIoU': val_mIoU}
            )

            if counter > PATIENCE:
                print('Early Stopping...')
                break
        
        if args.scheduler:
            scheduler.step()  

    save_table("Predictions", model, val_loader, device)
   

def validation(model, data_loader, device, criterion, epoch, args):
    print(f'Start validation!')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        # metric의 묶음을 한 번에 사용
        metric_collection = MetricCollection({
            "micro": MulticlassJaccardIndex(num_classes=n_class, average="micro"),
            "macro": MulticlassJaccardIndex(num_classes=n_class, average="macro"),      # mIoU
            "classwise": MulticlassJaccardIndex(num_classes=n_class, average="none")    # classwise IoU
        })
        metric_collection.cuda()

        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            metric_collection.update(outputs, masks)
        
        result = metric_collection.compute()
        micro = result["micro"].item()
        macro = result["macro"].item()
        classwise_results = result["classwise"].detach().cpu().numpy()
        category_list = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
                'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        IoU_by_class = [{classes : round(IoU, 4)} for IoU, classes in zip(classwise_results, category_list)]

        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch} || Average Loss: {round(avrg_loss.item(), 4)} || macro : {round(macro, 4)} || micro: {round(micro, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, macro, IoU_by_class

if __name__ == "__main__":

    dataset_path = '/opt/ml/input/data'
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["train"])

    print(args)
    seed_everything(args.seed)

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    CFG = {
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "seed" : args.seed,
        "encoder" : args.encoder,
        "encoder_weights" : args.encoder_weights,
        "decoder" : args.decoder,
        "optimizer" : args.optimizer,
        "scheduler" : args.scheduler,
        "criterion" : args.criterion,
    }

    wandb.init(
        project=args.project, entity=args.entity, name=args.experiment_name, config=CFG,
    )

    wandb.define_metric("Tr Loss", summary="min")
    wandb.define_metric("Tr mIoU", summary="max")

    wandb.define_metric("Val Loss", summary="min")
    wandb.define_metric("Val mIoU", summary="max")

    train(args)

    wandb.finish()

