from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split

# Agriculture Dataset ---------------------------------------------------------------------------
class SourceAndTargetDataset(Dataset):
    def __init__(self, source_img_paths, target_img_paths, transform, phase="train"):
        self.source_img_paths = source_img_paths
        self.target_img_paths = target_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.source_img_paths), len(self.target_img_paths)])

    def __getitem__(self, idx):
        source_img = Image.open(self.source_img_paths[idx])
        target_img = Image.open(self.target_img_paths[idx])

        # apply preprocessing transformations
        source_img = self.transform(source_img, self.phase)
        target_img = self.transform(target_img, self.phase)

        return {"source": source_img, "target": target_img}


# Data Module
class SourceAndTargetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, domainA, domainB, set_size=2000):
        super(SourceAndTargetDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.domainA = domainA
        self.domainB = domainB
        self.set_size = set_size

    def prepare_data(self):
        self.source_img_paths_train = glob.glob(
            os.path.join(self.data_dir, self.domainA, "satellite", "usa", "*.png")
        )
        self.target_img_paths_train = glob.glob(
            os.path.join(self.data_dir, self.domainB, "*.png")
        )

        self.source_img_paths_train.sort()
        self.target_img_paths_train.sort()
        print(self.source_img_paths_train[0])
        print(self.target_img_paths_train[0])
        assert self.set_size+100 <= len(self.source_img_paths_train) and self.set_size+100 <= len(self.target_img_paths_train)
        c = list(zip(self.source_img_paths_train, self.target_img_paths_train))
        random.Random(5).shuffle(c)
        a, b = zip(*c)
        self.source_img_paths_test = a[self.set_size:self.set_size+100]
        self.target_img_paths_test = b[self.set_size:self.set_size+100]
        self.source_img_paths_train = a[:self.set_size]
        self.target_img_paths_train = b[:self.set_size]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = SourceAndTargetDataset(
                self.source_img_paths_train, self.target_img_paths_train, self.transform, "train"
            )

        # Assign test dataset for use in dataloader(s)
        self.test_dataset = SourceAndTargetDataset(
                self.source_img_paths_test, self.target_img_paths_test, self.transform, "test"
            )

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