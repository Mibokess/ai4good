import os
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint

from data import CycleGAN_datamodule

from segmentation_system import SemanticSegmentationSystem
from pl_bolts.models.vision import UNet

pl.seed_everything(42)

num_cpus = 8

user = ''

net_type = "UNet_source_to_target"
os.makedirs(f"/cluster/scratch/{user}/logs/{net_type}", exist_ok=True)
epochs = 250
epoch_decay = epochs // 2
trn_batch_sz = num_cpus
tst_batch_sz = 32
num_workers = num_cpus

num_classes = 1

datamodule = CycleGAN_datamodule(
    num_workers=num_workers, 
    source_folder_name="predict_source", 
    mask_name='buildings',
    target_folder_name=None, 
    trn_batch_sz=trn_batch_sz, 
    tst_batch_sz=tst_batch_sz,
    multi_channel=True if num_classes > 1 else False
)

model = UNet(num_classes, 4, features_start=32)
system = SemanticSegmentationSystem(model, datamodule)

project_name = "unet"
log_path = f"/cluster/scratch/{user}/logs/{net_type}/"
wandb_logger = pl_loggers.WandbLogger(project=project_name, id=net_type, save_dir=log_path)
lr_logger = LearningRateMonitor(logging_interval = 'epoch')
checkpoint_callback = ModelCheckpoint(f'/cluster/scratch/{user}/logs/lightning_logs/{net_type}/', monitor = "validation_f1", save_top_k = 3, save_last = True, mode='max')
callbacks = [lr_logger, checkpoint_callback]

trainer = pl.Trainer(
    gpus = -1, 
    auto_select_gpus=True,
    max_epochs = epochs, 
    stochastic_weight_avg=True,
    progress_bar_refresh_rate = 50,
    precision = 16, 
    callbacks = callbacks, 
    benchmark=True,
    num_sanity_val_steps = 1, 
    logger=wandb_logger, 
    limit_train_batches=0.1,
    check_val_every_n_epoch=10
)

trainer.fit(system)
