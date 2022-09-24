from pytorch_lightning import LightningModule
import pytorch_lightning as pl

import torchvision.io

from torch.nn import functional as F
import torch

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.io import ImageReadMode
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchmetrics as tm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import random
from torchvision.utils import draw_segmentation_masks
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
from kornia.losses import dice_loss
from kornia.augmentation import Denormalize
import wandb

class SemanticSegmentationSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, datamodule: pl.LightningDataModule = None, lr: float = 1e-3, batch_size: int = 8):        
        super().__init__()
        self.model = model    
        self.datamodule = datamodule
        
        self.lr = lr
        self.batch_size = batch_size
        
        self.dice_loss = DiceLoss()
        self.IoU = tm.IoU(2)

        self.denormalize = Denormalize(0.5, 0.5)

    def forward(self, X, predict=False):
        y_pred = self.model(X.float())

        return y_pred
        
    def training_step(self, batch, batch_idx):
        X, y = batch['source_image'], batch['source_mask']
        
        X = X.float()
        y = y.float()
        
        y_pred = self(X)

        loss = self.dice_loss(y_pred, y) + nn.functional.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')

        self.log('training_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch['source_image'], batch['source_mask']
                
        X = X.float()
        y = y.int()
   
        y_pred = self(X)
        y_sig = torch.sigmoid(y_pred)
    
        accuracy = tm.functional.accuracy(y_sig, y)
        f1 = tm.functional.f1(y_sig, y)
        iou = self.IoU(y_sig, y)
        
        self.log('validation_accuracy', accuracy, prog_bar=True)
        self.log('validation_iou', iou, prog_bar=True)
        self.log('validation_f1', f1, prog_bar=True)
                
        return f1

    def validation_epoch_end(self, outputs):
        self.visualize_results_overlay(1)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
           
    @torch.no_grad()
    def visualize_results_overlay(self, num_batches=1):
        val_iterator = iter(self.val_dataloader())

        for i in range(num_batches):
            batch = next(val_iterator)
            Xs, ys = batch['source_image'], batch['source_mask']
                    
            y_preds = torch.sigmoid(self(Xs.float().cuda()))

            Xs = self.denormalize(Xs)[:, :3]

            mask_images = [wandb.Image(Xs[i].cpu().numpy().transpose(1, 2, 0), masks={
                "predictions": {
                    "mask_data": y_preds[i].round().long().sum(0).cpu().numpy(),
                     "class_labels": { 1: 'predictions'}
                },
                "ground_truth": {
                    "mask_data": ys[i].round().long().sum(0).cpu().numpy(),
                    "class_labels": { 1: 'ground_truth' }
                }
            }) for i in range(Xs.shape[0])]

            wandb.log({
               'mask_images': mask_images
            })

            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6, verbose=0)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'validation_f1'
        }

# from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice