from datetime import datetime
import uuid
import wandb
from pytorch_lightning.loggers import WandbLogger
from preprocessing.img_transforms import ImageTransform
from preprocessing.seg_transforms import SegImageTransform
from models.generators import CycleGANGenerator
from models.discriminators import CycleGANDiscriminator
from utils.weight_initializer import init_weights
from systems.cycle_gan_system import CycleGANSystem
from models.unet_light import UnetLight
from systems.seg_system_jaccard import SegmentationSystem
from os import path
import segmentation_models_pytorch as smp

from datasets.styletransfer import StyleTransferDataModule
from systems.style_test_system import StyleSegmentationTestOnlySystem
from torchvision.utils import make_grid
import numpy as np

from configs.pseudo_config import command_line_parser

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
    lr = {
        'G': 0.0002,
        'D': 0.0002
    }
    reconstr_w = 10
    id_w = 2

    # Data Preparation  -----------------------------------------------------------------
    transform = ImageTransform(img_size=256)
    label_transform = SegImageTransform(img_size=256)
    wandb.init(
            reinit=True,
            name=run_name,
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )

    # DataModule  ----------------------------------------------------------------- 
    G_basestyle = CycleGANGenerator(in_channels=4, out_channels=4, filter=32)
    G_stylebase = CycleGANGenerator(in_channels=4, out_channels=4, filter=32)
    D_base = CycleGANDiscriminator(in_channels=4, filter=32)
    D_style = CycleGANDiscriminator(in_channels=4,filter=32)

    # Init Weight  --------------------------------------------------------------
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type='normal')

    # LightningModule  --------------------------------------------------------------
    #model = CycleGANSystem(G_basestyle, G_stylebase, D_base, D_style, lr, transform, reconstr_w, id_w)
    cycle_config = {
        "G_s2t": G_basestyle,
        "G_t2s": G_stylebase,
        "D_source": D_base,
        "D_target": D_style,
        "lr": lr,
        "transform": transform,
        "reconstr_w": reconstr_w,
        "id_w": id_w,
    }

    if cfg.use_unet_light:
        net = UnetLight()
    else:
        net = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
    seg_config = {
        "net": net,
    }

    cycleGANSystem = CycleGANSystem.load_from_checkpoint(cfg.cycleGAN_checkpoint, **cycle_config)
    generator = cycleGANSystem.G_t2s
    segSystem = SegmentationSystem.load_from_checkpoint(cfg.seg_checkpoint, **seg_config)
    seg_net = segSystem.net

    dm = StyleTransferDataModule(generator, seg_net, data_dir, transform, label_transform, batch_size, set_size)

    dm.prepare_data() 
    dm.setup()
    dataloader = dm.test_dataloader()

    for batch in iter(dataloader):
        base, style, label_from_input, label_from_generated = (batch['base'], batch['style'], batch['label from input'], batch['label from generated'])

        base_img = make_grid(base[:,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().numpy()
        base_img = base_img/np.amax(base_img)

        style_img = make_grid(style[:,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().numpy()
        style_img = style_img/np.amax(style_img)

        label_input_img = make_grid(label_from_input, nrow=3, padding=2).permute(1, 2, 0).detach().numpy().astype(int)
        label_input_img = label_input_img * 255.0
        label_input_img = label_input_img.astype(int)

        label_generated_img = make_grid(label_from_generated, nrow=3, padding=2).permute(1, 2, 0).detach().numpy().astype(int)
        label_generated_img = label_generated_img * 255.0
        label_generated_img = label_generated_img.astype(int)

        wandb.log(
            {"B/S/BL&SL": [wandb.Image(base_img), wandb.Image(style_img), wandb.Image(label_input_img), wandb.Image(label_generated_img)]},
            commit=True,
        )


    wandb.finish()

if __name__ == "__main__":
    main()