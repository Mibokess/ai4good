from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split

# Buildings Dataset ---------------------------------------------------------------------------
class BuildingsDataset(Dataset):
    def __init__(self, color_img_paths, label_img_paths, transform, phase="train"):
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
        color_img, label_img = self.transform(color_img, label_img, self.phase)
        label_img = torch.where(label_img > 0, 1, 0)
        return {"color": color_img, "label": label_img}


# Data Module
class BuildingsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, set_size):
        super(BuildingsDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.set_size = set_size

    def prepare_data(self):
        self.color_img_paths_train = glob.glob(
            os.path.join(self.data_dir, "satellite", "usa", "*.png")
        )
        self.label_img_paths_train = glob.glob(
            os.path.join(self.data_dir, "masks", "buildings", "*.png")
        )

        self.color_img_paths_train.sort()
        self.label_img_paths_train.sort()
        print(self.color_img_paths_train[0])
        print(self.label_img_paths_train[0])
        #assert self.color_img_paths_train[0] == self.label_img_paths_train[0]
        c = list(zip(self.color_img_paths_train, self.label_img_paths_train))
        #random.Random(4).shuffle(c)
        random.Random(5).shuffle(c)
        a, b = zip(*c)
        self.color_img_paths_val = a[self.set_size:self.set_size+100]
        self.label_img_paths_val = b[self.set_size:self.set_size+100]
        self.color_img_paths_train = a[:self.set_size]
        self.label_img_paths_train = b[:self.set_size]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BuildingsDataset(
                self.color_img_paths_train, self.label_img_paths_train, self.transform, "train"
            )

        self.val_dataset = BuildingsDataset(
                self.color_img_paths_val, self.label_img_paths_val, self.transform, "test"
            )
    
    def get_test_paths(self):
        return (self.color_img_paths_val, self.label_img_paths_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=40,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=40
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=40
        )

