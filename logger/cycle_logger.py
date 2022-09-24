import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
import wandb


class CycleImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(self, data_module, log_key="Media/Pipeline", num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self.log_key = log_key

        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        val_samples = next(iter(dataloader))

        self.source_imgs = val_samples["source"]
        self.target_imgs = val_samples["target"]

    def on_train_epoch_end(self, trainer, pl_module, *args):
        source_imgs = self.source_imgs.to(device=pl_module.device)
        target_imgs = self.target_imgs.to(device=pl_module.device)

        # get the segmentation network
        G_s2t = pl_module.G_s2t
        G_t2s = pl_module.G_t2s

        generated_target = G_s2t(source_imgs)
        cycled_source = G_t2s(G_s2t(source_imgs))   

        generated_source = G_t2s(target_imgs)
        cycled_target = G_s2t(G_t2s(target_imgs))

        source = make_grid(source_imgs[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        source = source/np.amax(source)   
        source_2_target = make_grid(generated_target[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        source_2_target = source_2_target/np.amax(source_2_target) 
        source_cycled = make_grid(cycled_source[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        source_cycled = source_cycled/np.amax(source_cycled)  

        target = make_grid(target_imgs[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        target = target/np.amax(target)
        target_2_source = make_grid(generated_source[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        target_2_source = target_2_source/np.amax(target_2_source) 
        target_cycled = make_grid(cycled_target[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        target_cycled = target_cycled/np.amax(target_cycled)  

        try:
            # Log the images as wandb Image
            trainer.logger.experiment.log(
                {"source / target2source / source cycled": [wandb.Image(source), wandb.Image(target_2_source), wandb.Image(source_cycled)]},
                commit=False,
            )

            trainer.logger.experiment.log(
                {"target / source2taret / target cycled": [wandb.Image(target), wandb.Image(source_2_target), wandb.Image(target_cycled)]},
                commit=False,
            )

        except BaseException as err:
            print(f"Error occured while uploading image to wandb. {err=}, {type(err)=}")