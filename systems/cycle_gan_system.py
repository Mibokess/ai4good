import torch
from torchvision.utils import make_grid
from torch import nn, optim
import pytorch_lightning as pl


class CycleGANSystem(pl.LightningModule):
    def __init__(
        self,
        G_s2t,  # generator source to target
        G_t2s,  # generator target to source
        D_source,
        D_target,
        lr,
        transform,  # preprocessing transformation
        reconstr_w=10,  # reconstruction weighting
        id_w=2,  # identity weighting
        viz_set=None,
    ):
        super(CycleGANSystem, self).__init__()
        self.G_s2t = G_s2t
        self.G_t2s = G_t2s
        self.D_source = D_source
        self.D_target = D_target
        self.lr = lr
        self.transform = transform
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.g_s2t_optimizer = optim.Adam(
            self.G_s2t.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.g_t2s_optimizer = optim.Adam(
            self.G_t2s.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.d_source_optimizer = optim.Adam(
            self.D_source.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.d_target_optimizer = optim.Adam(
            self.D_target.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )

        return [
            self.g_s2t_optimizer,
            self.g_t2s_optimizer,
            self.d_source_optimizer,
            self.d_target_optimizer,
        ], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        source_img, target_img = (batch["source"], batch["target"])

        b = source_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_source = self.generator_loss(
                self.D_source(self.G_t2s(target_img)), valid
            )
            val_target = self.generator_loss(
                self.D_target(self.G_s2t(source_img)), valid
            )
            val_loss = (val_source + val_target) / 2

            # Reconstruction
            reconstr_source = self.mae(self.G_t2s(self.G_s2t(source_img)), source_img)
            reconstr_target = self.mae(self.G_s2t(self.G_t2s(target_img)), target_img)
            reconstr_loss = (reconstr_source + reconstr_target) / 2

            # Identity
            id_source = self.mae(self.G_t2s(source_img), source_img)
            id_target = self.mae(self.G_s2t(target_img), target_img)
            id_loss = (id_source + id_target) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            logs = {
                "G_loss": G_loss,
                "val_source": val_source,
                "val_target": val_target,
                "val_loss": val_loss,
                "reconstr_source": reconstr_source,
                "reconstr_target": reconstr_target,
                "reconstr_loss": reconstr_target,
                "id_source": reconstr_target,
                "id_target": id_target,
                "id_loss": id_loss,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return {
                "loss": G_loss,
                "validity": val_loss,
                "reconstr": reconstr_loss,
                "identity": id_loss,
            }

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_source_gen_loss = self.discriminator_loss(
                self.D_source(self.G_t2s(target_img)), fake
            )
            D_target_gen_loss = self.discriminator_loss(
                self.D_target(self.G_s2t(source_img)), fake
            )
            D_source_valid_loss = self.discriminator_loss(
                self.D_source(source_img), valid
            )
            D_target_valid_loss = self.discriminator_loss(
                self.D_target(target_img), valid
            )

            D_gen_loss = (D_source_gen_loss + D_target_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_source_valid_loss + D_target_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            logs = {
                "D_loss": D_loss,
                "D_source_gen_loss": D_source_gen_loss,
                "D_target_gen_loss": D_target_gen_loss,
                "D_source_valid_loss": D_source_valid_loss,
                "D_target_valid_loss": D_target_valid_loss,
                "D_gen_loss": D_gen_loss,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return {"loss": D_loss}

    def training_epoch_end(self, outputs):
        self.step += 1

        avg_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 4
                for i in range(4)
            ]
        )
        G_mean_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        D_mean_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2
                for i in [2, 3]
            ]
        )
        validity = sum(
            [
                torch.stack([x["validity"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        reconstr = sum(
            [
                torch.stack([x["reconstr"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        identity = sum(
            [
                torch.stack([x["identity"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )

        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)

        return None

    def generate(self, inputs):
        return self.G_s2t(inputs)