import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
from PIL import Image
from matplotlib import cm

import wandb

class VersusSemsegImageLogger(Callback):
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
        self.pseudo_label_imgs = val_samples["pseudo label"]
        self.pseudo_label_imgs = torch.unsqueeze(self.pseudo_label_imgs, dim=1)


    def on_train_epoch_end(self, trainer, pl_module, *args):
        rgb = self.rgb_imgs.to(device=pl_module.device)
        label = self.label_imgs.to(device=pl_module.device)
        pseudo_label = self.pseudo_label_imgs.to(device=pl_module.device)
        seg_r = pl_module.net_r(rgb)
        seg_r = seg_r.to(device=pl_module.device)
        seg_f = pl_module.net_f(rgb)
        seg_f = seg_f.to(device=pl_module.device)

        temp1 = make_grid(rgb[:,0:3,:,:], nrow=8, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        temp1 = temp1/np.amax(temp1)    

        temp2 = make_grid(label, nrow=8, padding=2).permute(1, 2, 0).detach().cpu().numpy().astype(int)
        temp2 = temp2 * 255.0
        temp2 = temp2.astype(int)

        temp3 = make_grid(seg_r, nrow=8, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        temp3 = temp3 * 255.0
        temp3 = temp3.astype(int)
        
        t = np.concatenate([temp1, temp2, temp3], axis=0)
        im = Image.fromarray(np.uint8(cm.gist_earth(t[:,:,0])*255))  

        temp2 = make_grid(pseudo_label, nrow=8, padding=2).permute(1, 2, 0).detach().cpu().numpy().astype(int)
        temp2 = temp2 * 255.0
        temp2 = temp2.astype(int)

        temp3 = make_grid(seg_f, nrow=8, padding=2).permute(1, 2, 0).detach().cpu().numpy()
        temp3 = temp3 * 255.0
        temp3 = temp3.astype(int)
        
        t2 = np.concatenate([temp1, temp2, temp3], axis=0)
        im2 = Image.fromarray(np.uint8(cm.gist_earth(t2[:,:,0])*255))

        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {"End of epoch results": [wandb.Image(im), wandb.Image(im2)]},
            commit=False,
        )