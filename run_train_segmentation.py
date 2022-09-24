from datetime import datetime
import uuid
from preprocessing.seg_transforms import SegImageTransform
from datasets.buildings import BuildingsDataModule
from datasets.streets import StreetsDataModule
from models.unet_light import UnetLight
import segmentation_models_pytorch as smp
from pytorch_lightning import Trainer

from systems.seg_system_jaccard import SegmentationSystem
from pytorch_lightning.loggers import WandbLogger, wandb
import wandb
from logger.image_logger import SemsegImageLogger

from configs.seg_config import command_line_parser

from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"{cfg.log_dir}/{run_name}/"
    data_dir = cfg.dataset_root

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    epoch = cfg.num_epochs_seg
    set_size = cfg.set_size

    # Data Preparation  -----------------------------------------------------------------
    transform = SegImageTransform(img_size=256)
    wandb.init(
            reinit=True,
            name=run_name,
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )

    # System Model Configuration  -----------------------------------------------------------------
    if cfg.use_builing_labels:
        dm = BuildingsDataModule(data_dir, transform, batch_size, set_size)
    else: 
        dm = StreetsDataModule(data_dir, transform, batch_size, set_size)

    if cfg.use_unet_light:
        net = UnetLight()
    else:
        net = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

    # LightningModule  --------------------------------------------------------------
    model = SegmentationSystem(net)

    # Logger  --------------------------------------------------------------
    seg_wandb_logger = (
        WandbLogger(project=project_name, name=run_name, prefix="seg")
        if cfg.use_wandb
        else None
    )

    # Callbacks  --------------------------------------------------------------
    # save the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        save_last=False,
        save_top_k=3,
        verbose=False,
        monitor="loss_val",
        mode="min",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    seg_image_callback = SemsegImageLogger(dm)

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    trainer = Trainer(
        max_epochs=epoch,
        gpus=1,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=seg_wandb_logger if cfg.use_wandb else None,
        callbacks=[checkpoint_callback, seg_image_callback],
    )

    # Train
    print("Fitting", run_name)
    trainer.fit(model, datamodule=dm)

    wandb.finish()

if __name__ == "__main__":
    main()


