from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split

# Streets Dataset ---------------------------------------------------------------------------
class GeneratedSegDataset(Dataset):
    def __init__(self, generator, color_img_paths, label_img_paths, transform, phase="train"):
        self.generator = generator
        self.color_img_paths = color_img_paths
        self.label_img_paths = label_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.color_img_paths), len(self.label_img_paths)])

    def __getitem__(self, idx):
        color_img = Image.open(self.color_img_paths[idx])
        label_img = Image.open(self.label_img_paths[idx])
        assert color_img.size == label_img.size
        
        # apply preprocessing transformations
        color, label = self.transform(color_img, label_img, self.phase)
        shape = color.shape

        with torch.no_grad():
            input = torch.reshape(color, (1, shape[0], shape[1], shape[2]))
            #generated = self.generator(input)
            #generated, _ = self.transform(generated, label_img, "pseudo")
            #generated = torch.squeeze(generated, 0)

        label = torch.where(label > 0, 1, 0)
        return {"color": color, "label": label }

# Data Module
class GeneratedSegDataModule(pl.LightningDataModule):
    def __init__(self, generator, data_dir, transform, batch_size, set_size):
        super(GeneratedSegDataModule, self).__init__()
        self.generator = generator
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.set_size = set_size
        self.color_img_paths_test = []
        self.label_img_paths_test = []

    def prepare_data(self):
        self.color_img_paths_train = glob.glob(
            os.path.join(self.data_dir, "satellite", "usa", "*.png")
        )
        self.label_img_paths_train = glob.glob(
            os.path.join(self.data_dir, "masks", "buildings", "*.png")
        )

        self.color_img_paths_train.sort()
        self.label_img_paths_train.sort()
        c = list(zip(self.color_img_paths_train, self.label_img_paths_train))
        random.Random(4).shuffle(c)
        a, b = zip(*c)
        self.color_img_paths_test = a[self.set_size:self.set_size+100]
        self.label_img_paths_test = b[self.set_size:self.set_size+100]
        self.color_img_paths_train = a[:self.set_size]
        self.label_img_paths_train = b[:self.set_size]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = GeneratedSegDataset(
            self.generator,
            self.color_img_paths_train, 
            self.label_img_paths_train, 
            self.transform, 
            "train"
        )

        self.test_dataset = GeneratedSegDataset(
            self.generator,
            self.color_img_paths_test, 
            self.label_img_paths_test, 
            self.transform, 
            "test"
        )
    
    def get_test_paths(self):
        return (self.color_img_paths_test, self.label_img_paths_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=40
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=40
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=40
        )