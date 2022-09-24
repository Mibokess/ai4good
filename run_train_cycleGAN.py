from datetime import datetime
import uuid
from pytorch_lightning.loggers.base import DummyExperiment
import wandb
from pytorch_lightning.loggers import WandbLogger
from preprocessing.img_transforms import ImageTransform
from preprocessing.seg_transforms import SegImageTransform
from datasets.sourcetarget import SourceAndTargetDataModule
from models.generators import CycleGANGenerator
from models.discriminators import CycleGANDiscriminator
from utils.weight_initializer import init_weights
from systems.cycle_gan_system import CycleGANSystem
from models.unet_light import UnetLight
from systems.seg_system_jaccard import SegmentationSystem
from logger.cycle_logger import CycleImageLogger
from os import path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from datasets.pseudo import PseudoLabelDataModule
import torch
from configs.cyclegan_config import command_line_parser


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"{cfg.log_dir}/{run_name}/"
    data_dir = cfg.dataset_root

    data_dir_domainA = cfg.domainA_dir
    data_dir_domainB = cfg.domainB_dir

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    epoch = cfg.num_epochs
    set_size = cfg.set_size
    reconstr_w = 10
    id_w = 2

    # Data Preparation  -----------------------------------------------------------
    transform = ImageTransform(img_size=256)
    wandb.init(
            reinit=True,
            name=run_name,
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )

    # DataModule  -----------------------------------------------------------------
    dm = SourceAndTargetDataModule(data_dir, transform, batch_size, set_size=set_size, domainA=data_dir_domainA, domainB=data_dir_domainB)

    G_basestyle = CycleGANGenerator(in_channels=4, out_channels=4, filter=32)
    G_stylebase = CycleGANGenerator(in_channels=4, out_channels=4, filter=32)
    D_base = CycleGANDiscriminator(in_channels=4, filter=32)
    D_style = CycleGANDiscriminator(in_channels=4, filter=32)

    # Init Weight  --------------------------------------------------------------
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type='normal')

    # LightningModule  --------------------------------------------------------------
    cycle_config = {
        "G_s2t": G_basestyle,
        "G_t2s": G_stylebase,
        "D_source": D_base,
        "D_target": D_style,
        "lr": {
            'G': 0.0002,
            'D': 0.0002
        },
        "transform": transform,
        "reconstr_w": reconstr_w,
        "id_w": id_w,
    }
    model = CycleGANSystem(**cycle_config)

    # Logger  --------------------------------------------------------------
    wandb_logger = (
        WandbLogger(project=project_name, name=run_name, prefix="cycleGAN")
        if cfg.use_wandb
        else None
    )

    # Callbacks  --------------------------------------------------------------
    # save the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(log_path, "cycleGAN"),
    )
    # wandb callback for images
    cycle_callback = CycleImageLogger(dm, log_key="Pipeline")


    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    trainer = Trainer(
        max_epochs=epoch,
        gpus=1,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0,  # Skip Sanity Check
        logger=wandb_logger if cfg.use_wandb else None,
        callbacks=[checkpoint_callback, cycle_callback],
    )

    # Train
    print("Fitting", run_name)
    trainer.fit(model, datamodule=dm)

    wandb.finish()

if __name__ == "__main__":
    main()