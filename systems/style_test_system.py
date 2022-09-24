import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim

from models.jaccard import IoULoss
from torchmetrics import IoU

import pytorch_lightning as pl

class StyleSegmentationTestOnlySystem(pl.LightningModule):
    def __init__(self, net, n_classes=1):
        super(StyleSegmentationTestOnlySystem, self).__init__()
        self.net = net

    def training_step(self, batch, batch_nb, optimizer_idx):
        pass
        
    def configure_optimizers(self):
        pass

    def validation_step(self, batch, batch_nb):
        pass

    def test_step(self, batch, batch_idx):
        pass
