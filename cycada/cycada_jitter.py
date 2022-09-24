from models import *
import pytorch_lightning as pl
import itertools
from torch import optim
import numpy as np
import torchvision
import wandb
from segmentation_system import SemanticSegmentationSystem
from torchvision.transforms import Normalize 
from kornia.augmentation import ColorJitter as CJ

from losses import Loss
from pl_bolts.models.vision import UNet

class Cycada(pl.LightningModule):
    def __init__(
        self, 
        d_lr: float = 2e-4, 
        g_lr: float = 2e-4, 
        beta_1: float = 0.5, 
        beta_2: float = 0.999, 
        epoch_decay: int = 200, 
        net_type='resnet', 
        datamodule=None, 
        num_workers: int = 4,
        batch_size: int = 4, 
        seg_system_checkpoint_path=None, 
        filter_size=32,
        num_classes=1,
        user='user'
    ):

        super().__init__()

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch_decay = epoch_decay
        self.loss_change = epoch_decay // 5

        self.fake_pool_A = ImagePool(pool_sz = 50)
        self.fake_pool_B = ImagePool(pool_sz = 50)

        self.loss = Loss(loss_type = 'MSE', lambda_ = 10)
        init = Initializer(init_type = 'normal', init_gain = 0.02)

        if net_type == 'resnet':
            self.d_A = init(Discriminator(in_channels = 4, out_channels = filter_size, nb_layers = 3))
            self.d_B = init(Discriminator(in_channels = 4, out_channels = filter_size, nb_layers = 3))
            self.g_A2B = init(Generator(in_channels = 4, out_channels = filter_size, apply_dp = False))
            self.g_B2A = init(Generator(in_channels = 4, out_channels = filter_size, apply_dp = False))
        else:
            self.d_A = init(CycleGANDiscriminator(filter_size))
            self.d_B = init(CycleGANDiscriminator(filter_size))
            self.g_A2B = init(CycleGANGenerator(filter_size))
            self.g_B2A = init(CycleGANGenerator(filter_size))
            
        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params   = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])

        self.datamodule = datamodule

        self.normalize = Normalize([0.5] * 4, [0.5] * 4)

        if seg_system_checkpoint_path is not None:
            self.seg_system = SemanticSegmentationSystem.load_from_checkpoint(seg_system_checkpoint_path, model=UNet(num_classes, 4, features_start=32))

        self.example_input_array = [torch.rand(1, 4, img_sz, img_sz, device = self.device),
                                    torch.rand(1, 4, img_sz, img_sz, device = self.device)]

        self.user = user

    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def forward(self, real_A, real_B, noise=True):          
        fake_B = self.g_A2B(real_A, noise)
        fake_A = self.g_B2A(real_B, noise)

        return fake_B, fake_A
    
    def forward_gen(self, real_A, real_B, fake_A, fake_B, noise=True, jitter=False, y_A=None):
        if jitter:
            jitter = CJ(0, 0, .3, .8, p=0.8)
            fake_B_masked_jitter = torch.zeros_like(fake_B)
            fake_B_masked_jitter[:, :3] = jitter(fake_B[:, :3])
            fake_B_masked_jitter[:, 3] = fake_B[:, 3]  

            cyc_A = self.g_B2A(fake_B_masked_jitter, noise)
            cyc_B = self.g_A2B(fake_A, noise)
        else: 
            cyc_A = self.g_B2A(fake_B, noise)
            cyc_B = self.g_A2B(fake_A, noise)
        
        idt_A = self.g_B2A(real_A, noise)
        idt_B = self.g_A2B(real_B, noise)
        
        return cyc_A, idt_A, cyc_B, idt_B
       
    @staticmethod
    def forward_dis(dis, real_data, fake_data):
        pred_real_data = dis(real_data)
        pred_fake_data = dis(fake_data)
        
        return pred_real_data, pred_fake_data

    def forward_seg(self, real_A, fake_B, cyc_A):
        predict_real_A = torch.sigmoid(self.seg_system(real_A))
        predict_fake_B = torch.sigmoid(self.seg_system(self.normalize(fake_B)))
        predict_reconstructed_A = torch.sigmoid(self.seg_system(self.normalize(cyc_A)))
        
        return predict_real_A, predict_fake_B, predict_reconstructed_A

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B, y_A, y_B = batch['source_image'], batch['target_image'], batch['source_mask'], batch['target_mask']
        fake_B, fake_A = self(real_A, real_B)
        fake_B_no_noise, fake_A_no_noise = self(real_A, real_B, False)

        if optimizer_idx == 0:
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B, True, True, y_A)
            cyc_A_no_noise, _, _, _ = self.forward_gen(real_A, real_B, fake_A, fake_B, False, True, y_A)
            
            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B, self.seg_system], requires_grad = False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(fake_B)

            g_A2B_loss, g_B2A_loss, g_tot_loss, g_A2B_idt_loss, g_B2A_idt_loss, tot_cyc_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, 
                                                                        d_A_pred_fake_data, d_B_pred_fake_data)

            predict_real_A, predict_fake_B, predict_reconstructed_A = self.forward_seg(real_A, fake_B_no_noise, cyc_A_no_noise)
            seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A = self.loss.get_seg_loss(predict_real_A, predict_fake_B, predict_reconstructed_A, y_A)

            seg_loss = self.seg_loss_scaled(seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A)
            g_tot_loss += seg_loss

            dict_ = {
                'g_tot_train_loss': g_tot_loss, 
                f'seg_loss_scaled': seg_loss,
                'g_A2B_train_loss': g_A2B_loss, 
                'g_B2A_train_loss': g_B2A_loss,
                'g_A2B_idt_loss': g_A2B_idt_loss,
                'g_B2A_idt_loss': g_B2A_idt_loss,
                'seg_loss_real_A': seg_loss_real_A,
                'seg_loss_transformed_A': seg_loss_transformed_A,
                'predict_reconstructed_A': seg_loss_reconstructed_A
            }
            self.log_dict(dict_, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return g_tot_loss
    
        if optimizer_idx == 1:   
            self.set_requires_grad([self.d_A], requires_grad = True)
            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A.detach())

            # GAN loss
            d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
            self.log("d_A_train_loss", d_A_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return d_A_loss  

        if optimizer_idx == 2:   
            self.set_requires_grad([self.d_B], requires_grad = True)
            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B.detach())

            # GAN loss
            d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            self.log("d_B_train_loss", d_B_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return d_B_loss
        
        if optimizer_idx == 3 and self.current_epoch > 15:
            self.set_requires_grad([self.seg_system], requires_grad = True)
            cyc_A_no_noise, _, _, _ = self.forward_gen(real_A, real_B, fake_A, fake_B, False, True, y_A)

            predict_real_A, predict_fake_B, predict_reconstructed_A = self.forward_seg(real_A, fake_B_no_noise, cyc_A_no_noise)
            seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A = self.loss.get_seg_loss(predict_real_A, predict_fake_B, predict_reconstructed_A, y_A)

            seg_loss = self.seg_loss_scaled(seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A)

            self.log_dict({
                'seg_loss_real_A': seg_loss_real_A,
                'seg_loss_transformed_A': seg_loss_transformed_A,
                'seg_loss_reconstructed_A': seg_loss_reconstructed_A,
                f'seg_loss_scaled': seg_loss,
            }, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return seg_loss

    @staticmethod
    def seg_loss_scaled(seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A):
        return seg_loss_real_A +  0.5 * seg_loss_transformed_A +  2 * seg_loss_reconstructed_A

    def shared_step(self, batch, stage: str = 'val', log_images=False, samples_to_log=16):
        real_A, real_B, y_A, y_B = batch['source_image'], batch['target_image'], batch['source_mask'], batch['target_mask']
        
        fake_B, fake_A = self(real_A, real_B)
        fake_B_no_noise, fake_A_no_noise = self(real_A, real_B, False)

        cyc_A , idt_A , cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B, True, True, y_A)
        cyc_A_no_noise, _, cyc_B_no_noise, _ = self.forward_gen(real_A, real_B, fake_A, fake_B, False, True, y_A)

        predict_real_A, predict_fake_B, predict_reconstructed_A = self.forward_seg(real_A, fake_B_no_noise, cyc_A_no_noise)
        predict_real_B = torch.sigmoid(self.seg_system(real_B))
        predict_fake_A = torch.sigmoid(self.seg_system(self.normalize(fake_A_no_noise)))
            
        seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A = self.loss.get_seg_loss(predict_real_A, predict_fake_B, predict_reconstructed_A, y_A)

        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B)
        
        g_A2B_loss, g_B2A_loss, g_tot_loss, _, _, tot_cyc_loss= self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, 
                                                                    d_A_pred_fake_data, d_B_pred_fake_data)

        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        seg_loss = self.seg_loss_scaled(seg_loss_real_A, seg_loss_transformed_A, seg_loss_reconstructed_A)
        g_tot_loss += seg_loss

        dict_ = {
            f'g_tot_{stage}_loss': g_tot_loss,
            f'seg_{stage}_loss_scaled': seg_loss,
            f'g_A2B_{stage}_loss': g_A2B_loss, 
            f'g_B2A_{stage}_loss': g_B2A_loss, 
            f'd_A_{stage}_loss'  : d_A_loss  , 
            f'd_B_{stage}_loss'  : d_B_loss,
            f'seg_{stage}_loss_real_A': seg_loss_real_A,
            f'seg_{stage}_loss_transformed_A': seg_loss_transformed_A,
            f'seg_{stage}_loss_reconstructed_A': seg_loss_reconstructed_A
        }

        predict_reconstructed_B = torch.sigmoid(self.seg_system(self.normalize(cyc_B_no_noise)))

        if y_B != []:
            seg_loss_real_B, seg_loss_transformed_B, seg_loss_reconstructed_B = self.loss.get_seg_loss(predict_real_B, predict_fake_A, predict_reconstructed_B, y_B.to(self.device))

            dict_[f'seg_{stage}_loss_real_B'] = seg_loss_real_B
            dict_[f'seg_{stage}_loss_transformed_B'] = seg_loss_transformed_B
            dict_[f'seg_{stage}_loss_reconstructed_B'] = seg_loss_reconstructed_B

        self.log_dict(dict_, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        if log_images:
            idt_A_images = [wandb.Image(self.transform_to_numpy_image(idt_A[i])) for i in range(samples_to_log)]
            idt_B_images = [wandb.Image(self.transform_to_numpy_image(idt_B[i])) for i in range(samples_to_log)]

            predict_real_A = predict_real_A.round().long()
            predict_real_B = predict_real_B.round().long()
            predict_fake_B = predict_fake_B.round().long()
            predict_fake_A = predict_fake_A.round().long()
            predict_reconstructed_A = predict_reconstructed_A.round().long()
            predict_reconstructed_B = predict_reconstructed_B.round().long()

            predict_real_A = predict_real_A.sum(1).cpu().numpy()
            predict_real_B = predict_real_B.sum(1).cpu().numpy()
            predict_fake_B = predict_fake_B.sum(1).cpu().numpy()
            predict_fake_A = predict_fake_A.sum(1).cpu().numpy()
            predict_reconstructed_A = predict_reconstructed_A.sum(1).cpu().numpy()
            predict_reconstructed_B = predict_reconstructed_B.sum(1).cpu().numpy()
            y_A = y_A.sum(1).cpu().numpy()
            if y_B != []:
                y_B = y_B.sum(1).cpu().numpy()

            class_labels = { 1: 'prediction' }

            real_A_mask_images = [wandb.Image(self.transform_to_numpy_image(real_A[i]), masks={
                "predictions": {
                    "mask_data": predict_real_A[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_A[i],
                    "class_labels": class_labels
                }
            }) for i in range(samples_to_log)]

            fake_B_mask_images = [wandb.Image(self.transform_to_numpy_image(fake_B[i]), masks={
                "predictions": {
                    "mask_data": predict_fake_B[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_A[i],
                    "class_labels": class_labels
                }
            }) for i in range(samples_to_log)]

            real_B_mask_images = [wandb.Image(self.transform_to_numpy_image(real_B[i]), masks={
                "predictions": {
                    "mask_data": predict_real_B[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_B[i],
                    "class_labels": class_labels
                } if y_B != [] else None
            }) for i in range(samples_to_log)]

            fake_A_mask_images = [wandb.Image(self.transform_to_numpy_image(fake_A[i]), masks={
                "predictions": {
                    "mask_data": predict_fake_A[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_B[i],
                    "class_labels": class_labels
                } if y_B != [] else None
            }) for i in range(samples_to_log)]

            cyc_A_mask_images = [wandb.Image(self.transform_to_numpy_image(cyc_A[i]), masks={
                "predictions": {
                    "mask_data": predict_reconstructed_A[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_A[i],
                    "class_labels": class_labels
                }
            }) for i in range(samples_to_log)]

            cyc_B_mask_images = [wandb.Image(self.transform_to_numpy_image(cyc_B[i]), masks={
                "predictions": {
                    "mask_data": predict_reconstructed_B[i],
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": y_B[i],
                    "class_labels": class_labels
                } if y_B != [] else None
            }) for i in range(samples_to_log)]

            wandb.log({
                "idt_A_images": idt_A_images,
                "idt_B_images": idt_B_images,
                "real_A_mask_images": real_A_mask_images,
                "fake_A_mask_images": fake_A_mask_images,
                "fake_B_mask_images": fake_B_mask_images,
                "real_B_mask_images": real_B_mask_images,
                "cyc_A_mask_images": cyc_A_mask_images,
                "cyc_B_mask_images": cyc_B_mask_images
            })

    
    def transform_to_numpy_image(self, image):
        return image[:3].cpu().numpy().transpose(1, 2, 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        batch = next(iter(self.datamodule.val_dataloader()))
        batch['source_image'] = batch['source_image'].cuda()
        batch['source_mask'] = batch['source_mask'].cuda()
        batch['target_image'] = batch['target_image'].cuda()
        self.shared_step(batch, 'val', log_images=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def predict_step(self, batch, batch_idx):
        real_A, real_B, y_A, y_B = batch['source_image'], batch['target_image'], batch['source_mask'], batch['target_mask']
        
        fake_B, fake_A = self(real_A, real_B, True)

        os.makedirs(f'/cluster/scratch/{self.user}/predict_source/satellite/usa/', exist_ok=True)
        os.makedirs(f'/cluster/scratch/{self.user}/predict_source/masks/streets/', exist_ok=True)
        os.makedirs(f'/cluster/scratch/{self.user}/predict_source/masks/buildings/', exist_ok=True)

        for i, fake_B_image in enumerate(fake_B):
            save_image(fake_B_image, f'/cluster/scratch/{self.user}/predict_source/satellite/usa/{self.num_images_saved:05}.png', normalize=True)
            save_image(y_A[i], f'/cluster/scratch/{self.user}/predict_source/masks/streets/{self.num_images_saved:05}.png', normalize=True)
            save_image(y_B[i], f'/cluster/scratch/{self.user}/predict_source/masks/buildings/{self.num_images_saved:05}.png', normalize=True)
            self.num_images_saved += 1

        return fake_B

    def lr_lambda(self, epoch):
        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction

    def loss_lambda(self, epoch):
        fraction = (epoch - self.loss_change) / self.loss_change
        return 1 if epoch < self.loss_change else max(1, 1 - fraction)


    def configure_optimizers(self):
        g_opt   = optim.Adam(self.g_params  , lr = self.g_lr, betas = (self.beta_1, self.beta_2))
        d_A_opt = optim.Adam(self.d_A_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        d_B_opt = optim.Adam(self.d_B_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        seg_system_opt = optim.Adam(self.seg_system.parameters(), lr=1e-6, betas = (self.beta_1, self.beta_2))

        g_sch   = optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda = self.lr_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda = self.lr_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda = self.lr_lambda)
        
        return [g_opt, d_A_opt, d_B_opt, seg_system_opt], [g_sch, d_A_sch, d_B_sch]

    
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()
    
    def test_dataloader(self):
        return self.datamodule.test_dataloader()
