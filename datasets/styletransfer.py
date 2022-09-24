from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Streets Dataset ---------------------------------------------------------------------------
class StyleTransferDataset(Dataset):
    def __init__(self, generator, seg_net, base_img_paths, transform, label_transform, phase="train"):
        self.generator = generator
        self.seg_net = seg_net
        self.base_img_paths = base_img_paths
        self.transform = transform
        self.label_transform = label_transform
        self.phase = phase

    def __len__(self):
        return len(self.base_img_paths)

    def __getitem__(self, idx):
        base_img = Image.open(self.base_img_paths[idx])
        
        # apply preprocessing transformations
        base = self.transform(base_img, self.phase)
        shape = base.shape

        with torch.no_grad():
            input = torch.reshape(base, (1, shape[0], shape[1], shape[2]))

            seg_input = transforms.CenterCrop(128)(input)

            label_from_input = self.seg_net(seg_input)
            label_from_input = torch.reshape(label_from_input, (label_from_input.shape[1], label_from_input.shape[2], label_from_input.shape[3]))
            label_from_input = torch.squeeze(label_from_input, 0)
            generated = self.generator(input)

            generated = transforms.CenterCrop(128)(generated)

            label_from_generated = self.seg_net(generated)
            label_from_generated = torch.reshape(label_from_generated, (label_from_generated.shape[1], label_from_generated.shape[2], label_from_generated.shape[3]))
            label_from_generated = torch.squeeze(label_from_generated, 0)
            style = self.transform(generated, "generated")
            style = torch.squeeze(style, 0)

        _, label_from_input = self.label_transform(base, label_from_input, "pseudo")
        label_from_input = torch.clamp(label_from_input, min=0.0, max=1.0)
        label_from_input = label_from_input.int()
        label_from_input = torch.unsqueeze(label_from_input, dim=0)

        _, label_from_generated = self.label_transform(base, label_from_generated, "pseudo")
        label_from_generated = torch.clamp(label_from_generated, min=0.0, max=1.0)
        label_from_generated = label_from_generated.int()
        label_from_generated = torch.unsqueeze(label_from_generated, dim=0)

        base = transforms.CenterCrop(128)(base)

        return {"base": base, "style": style , "label from input": label_from_input, "label from generated": label_from_generated}

# Data Module
class StyleTransferDataModule(pl.LightningDataModule):
    def __init__(self, generator, seg_net, data_dir, transform, label_transform, batch_size, set_size):
        super(StyleTransferDataModule, self).__init__()
        self.generator = generator
        self.seg_net = seg_net
        self.data_dir = data_dir
        self.transform = transform
        self.label_transform = label_transform
        self.batch_size = batch_size
        self.set_size = set_size
        self.base_img_paths_test = []

    def prepare_data(self):
        self.base_img_paths_train = glob.glob(
            os.path.join(self.data_dir, "*.png")
        )

        self.base_img_paths_train.sort()
        random.Random(4).shuffle(self.base_img_paths_train)
        self.base_img_paths_test = self.base_img_paths_train[self.set_size:self.set_size+100]
        self.base_img_paths_train = self.base_img_paths_train[:self.set_size]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = StyleTransferDataset(
            self.generator,
            self.seg_net,
            self.base_img_paths_train, 
            self.transform, 
            self.label_transform,
            "train"
        )

        self.test_dataset = StyleTransferDataset(
            self.generator,
            self.seg_net,
            self.base_img_paths_test, 
            self.transform,
            self.label_transform, 
            "test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
