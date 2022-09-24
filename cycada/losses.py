import torch
from torch import nn
from kornia.losses import dice_loss

class Loss:

    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """

    def __init__(self, loss_type: str = 'MSE', lambda_: int = 10):

        """
        Parameters:
            loss_type: Loss Function to train CycleGAN
            lambda_:   Weightage of Cycle-consistency loss
        """

        self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        self.lambda_ = lambda_

        self.gen_loss_weight = 1
        self.cycle_loss_weight = 1
        self.identity_loss_weight = 0

        self.dice_loss = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()


    def get_dis_loss(self, dis_pred_real_data, dis_pred_fake_data):

        """
        Parameters:
            dis_pred_real_data: Discriminator's prediction on real data
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        dis_tar_real_data = torch.ones_like (dis_pred_real_data, requires_grad = False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad = False)

        loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self.loss(dis_pred_fake_data, dis_tar_fake_data)

        dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5

        return dis_tot_loss


    def get_gen_gan_loss(self, dis_pred_fake_data):

        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad = False)
        gen_tot_loss = self.loss(dis_pred_fake_data, gen_tar_fake_data)

        return gen_tot_loss


    def get_gen_cyc_loss(self, real_data, cyc_data):

        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """

        gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
        gen_tot_loss = gen_cyc_loss * self.lambda_

        return gen_tot_loss


    def get_gen_idt_loss(self, real_data, idt_data):

        """
        Implements the identity loss:
            nn.L1Loss(LG_B2A(real_A), real_A)
            nn.L1Loss(LG_A2B(real_B), real_B)
        """

        gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
        gen_tot_loss = gen_idt_loss * self.lambda_ * 0.5

        return gen_tot_loss


    def get_gen_loss(self, real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, d_A_pred_fake_data,
                     d_B_pred_fake_data):

        """
        Implements the total Generator loss
        Sum of Cycle loss, Identity loss, and GAN loss
        """

        #Cycle loss
        cyc_loss_A = self.get_gen_cyc_loss(real_A, cyc_A)
        cyc_loss_B = self.get_gen_cyc_loss(real_B, cyc_B)
        tot_cyc_loss = cyc_loss_A + cyc_loss_B

        # GAN loss
        g_A2B_gan_loss = self.get_gen_gan_loss(d_B_pred_fake_data)
        g_B2A_gan_loss = self.get_gen_gan_loss(d_A_pred_fake_data)

        # Identity loss
        g_B2A_idt_loss = self.get_gen_idt_loss(real_A, idt_A)
        g_A2B_idt_loss = self.get_gen_idt_loss(real_B, idt_B)

        # Total individual losses
        g_A2B_loss = self.gen_loss_weight * g_A2B_gan_loss + self.identity_loss_weight * g_A2B_idt_loss + self.cycle_loss_weight * tot_cyc_loss
        g_B2A_loss = self.gen_loss_weight * g_B2A_gan_loss + self.identity_loss_weight * g_B2A_idt_loss + self.cycle_loss_weight * tot_cyc_loss
        g_tot_loss = g_A2B_loss + g_B2A_loss - tot_cyc_loss

        return g_A2B_loss, g_B2A_loss, g_tot_loss, g_A2B_idt_loss, g_B2A_idt_loss, tot_cyc_loss

    def get_seg_loss(self, predict_real, predict_fake, predict_reconstructed, y_A):
        loss_real_A = self.dice_loss(predict_real, y_A) + nn.functional.binary_cross_entropy_with_logits(predict_real, y_A, reduction='mean')
        loss_transformed_A = self.dice_loss(predict_fake, y_A) + nn.functional.binary_cross_entropy_with_logits(predict_fake, y_A, reduction='mean')
        loss_reconstructed_A = self.dice_loss(predict_reconstructed, y_A) + nn.functional.binary_cross_entropy_with_logits(predict_reconstructed, y_A, reduction='mean')

        return loss_real_A, loss_transformed_A, loss_reconstructed_A

    def seg_loss(self, y_pred, y):
        bce_loss = self.cross_entropy(y_pred, y)
        dice_loss = dice_loss(y_pred, y)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice