import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
from PIL import Image
from matplotlib import cm

import wandb

class SemsegImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(self, data_module, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.test_dataloader()
        val_samples = next(iter(dataloader))

        self.rgb_imgs = val_samples["color"]
        self.label_imgs = val_samples["label"]
        self.label_imgs = torch.unsqueeze(self.label_imgs, dim=1)


    def on_train_epoch_end(self, trainer, pl_module, *args):
        rgb = self.rgb_imgs.to(device=pl_module.device)
        label = self.label_imgs.to(device=pl_module.device)
        seg = pl_module.net(rgb)
        seg = seg.to(device=pl_module.device)

        temp1 = make_grid(rgb[0:3,0:3,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        temp1 = temp1/np.amax(temp1)    

        temp2 = make_grid(label[0:3,:,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy().astype(int)
        temp2 = temp2 * 255.0
        temp2 = temp2.astype(int)

        temp3 = make_grid(seg[0:3,:,:,:], nrow=3, padding=2).permute(1, 2, 0).detach().cpu().numpy().astype(int)
        temp3 = temp3 * 255.0
        temp3 = temp3.astype(int)
        
        t = np.concatenate([temp2, temp3], axis=0)
        im = Image.fromarray(np.uint8(cm.gist_earth(t[:,:,0])*255))

        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {"End of epoch results": [wandb.Image(temp1), wandb.Image(im)]},
            commit=False,
        )