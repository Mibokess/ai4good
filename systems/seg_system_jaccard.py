import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim

from models.jaccard import IoULoss
from torchmetrics import IoU

import pytorch_lightning as pl

class SegmentationSystem(pl.LightningModule):
    def __init__(self, net, n_classes=1):
        super(SegmentationSystem, self).__init__()
        self.net = net
        self.seg_loss = IoULoss()
        self.iou_metric = IoU(num_classes=2)

    def training_step(self, batch, batch_nb):
        x, y = (batch['color'], batch['label'])
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y = torch.unsqueeze(y, dim=1) 
        y_hat = self.net(x)
        loss = self.seg_loss(y_hat, y.float())
        jaccard_index = self.iou_metric(y_hat, y.int())
        self.log_dict({
            'loss_train': loss,
            'jaccard_index_train': jaccard_index,
        }, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.999))
        return [optimizer], []

    def inference_step(self, batch):
        x = batch['color']
        if torch.cuda.is_available():
            x = x.cuda()
        y_hat = self.net(x)
        return y_hat
    
    def validation_step(self, batch, batch_nb):
        y_hat = self.inference_step(batch)
        y = batch['label']
        if torch.cuda.is_available():
            y = y.cuda()
        y = torch.unsqueeze(y, dim=1) 
        loss_val_unet = self.seg_loss(y_hat, y.float())
        jaccard_index = self.iou_metric(y_hat, y.int())
        self.log_dict({
            'loss_val': loss_val_unet,
            'jaccard_index_val': jaccard_index,
        }, on_step=False, on_epoch=True)


