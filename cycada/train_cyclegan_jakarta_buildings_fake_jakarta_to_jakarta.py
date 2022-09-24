import numpy as np
import os
import warnings, itertools, functools
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
pl.seed_everything(36)

from data_source import CycleGAN_datamodule
from cycada_jitter import *

num_cpus = 8

net_type = "cycada_fake_jakarta_to_jakarta"
os.makedirs(f"/cluster/scratch/{user}/logs/{net_type}", exist_ok=True)
epochs = 200
epoch_decay = epochs // 2
trn_batch_sz = 16
tst_batch_sz = 16
num_workers = num_cpus

num_classes = 1

datamodule = CycleGAN_datamodule(
    num_workers=num_workers, 
    source_folder_name="predict_source", 
    target_folder_name="splits_jakarta",
    target_root_dir='/cluster/scratch/jehrat/ai4good',
    target_satellite_name='satellite',
    mask_name='buildings',
    trn_batch_sz=trn_batch_sz, 
    tst_batch_sz=tst_batch_sz,
    multi_channel=True if num_classes > 1 else False
)

model = Cycada(
    epoch_decay = epoch_decay, 
    net_type=net_type, 
    datamodule=datamodule, 
    filter_size=32, 
    num_classes=num_classes,
    user=user,
    seg_system_checkpoint_path='/cluster/scratch/{user}/logs/lightning_logs/UNet_source_to_target/last.ckpt'
)

project_name = "cycada"
log_path = f"/cluster/scratch/{user}/logs/{net_type}/"
wandb_logger = pl_loggers.WandbLogger(project=project_name, id=net_type, save_dir=log_path)
lr_logger = LearningRateMonitor(logging_interval = 'epoch')
checkpoint_callback = ModelCheckpoint(f'/cluster/scratch/{user}/logs/lightning_logs/{net_type}/', monitor = "seg_val_loss_scaled", save_top_k = 3, save_last = True, mode='min')
callbacks = [lr_logger, checkpoint_callback]

trainer = pl.Trainer(
    gpus = -1, 
    max_epochs = epochs, 
    precision = 16, 
    callbacks = callbacks, 
    benchmark=True,
    num_sanity_val_steps = 1, 
    logger=wandb_logger, 
    limit_train_batches=0.02,
    #limit_val_batches=0.01,
    check_val_every_n_epoch=10
)

trainer.fit(model)
