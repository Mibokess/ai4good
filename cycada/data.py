import imageio as io
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pytorch_lightning as pl
from transforms import *
import os

class SourceTargetDataset(Dataset):

    def __init__(self, source_paths, target_paths, transforms=None, is_test=False, mask_name='streets', multi_channel=False, load_target_mask=False, load_second_mask_as_target=False):
        super().__init__()

        self.transforms = transforms

        self.source_paths = source_paths
        self.target_paths = target_paths

        self.is_test = is_test
        self.mask_name = mask_name
        self.multi_channel = multi_channel
        self.load_target_mask = load_target_mask
        self.load_second_mask_as_target = load_second_mask_as_target

    def __len__(self):
        return max(len(self.source_paths), len(self.target_paths)) if self.target_paths is not None else len(self.source_paths)

    def __getitem__(self, idx):
        source_image = io.imread(self.source_paths[idx % len(self.source_paths)])
        target_image = io.imread(self.target_paths[idx % len(self.target_paths)]) if self.target_paths is not None else None
        
        try:
            source_mask = io.imread(self.source_paths[idx % len(self.source_paths)].replace('/satellite/usa', f'/masks/{self.mask_name}'))[:, :, None]
        except:
            source_mask = None

        if self.is_test or self.load_target_mask:
            target_mask = io.imread(self.target_paths[idx % len(self.target_paths)].replace('/satellite/usa', f'/masks/{self.mask_name}'))[:, :, None]
        else:
            target_mask = None

        if self.load_second_mask_as_target:
            target_mask = io.imread(self.source_paths[idx % len(self.source_paths)].replace('/satellite/usa', f'/masks/buildings'))[:, :, None]

        sample = self.transforms({
            'source_image': source_image / 255, 
            'target_image': target_image / 255 if target_image is not None else None,

            'source_mask': (source_mask / 255).round() if source_mask is not None else None,
            'target_mask': (target_mask / 255).round() if target_mask is not None else None
        })

        return sample
        

class CycleGAN_datamodule(pl.LightningDataModule):
    def __init__(self, source_root_dir: str = "/cluster/scratch/mboss", target_root_dir: str = "/cluster/scratch/mboss", source_folder_name="splits_less", target_folder_name="nevada_less", source_satellite_name = 'satellite/usa', target_satellite_name = 'satellite/usa', img_sz: int = 256, trn_batch_sz: int = 4,
                 tst_batch_sz: int = 64, num_workers=4, multi_channel=False, mask_name='streets', load_second_mask_as_target=False, load_target_mask=False):
        super().__init__()

        self.source_root_dir = source_root_dir
        self.target_root_dir = target_root_dir
        self.source_folder_name = source_folder_name
        self.target_folder_name = target_folder_name
        self.target_satellite_name = target_satellite_name
        self.source_satellite_name = source_satellite_name

        self.trn_batch_sz = trn_batch_sz
        self.tst_batch_sz = tst_batch_sz

        self.transforms = {
            "train": T.Compose([To_Tensor(), Resize(img_sz), Random_Flip(), Normalize()]),
            "test": T.Compose([To_Tensor(), Resize(img_sz), Normalize()])
        }

        self.num_workers = num_workers
        self.mask_name = mask_name
        self.load_second_mask_as_target = load_second_mask_as_target
        self.multi_channel = multi_channel

        self.load_target_mask = load_target_mask

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        self.source_image_paths = [os.path.join(f'{self.source_root_dir}/{self.source_folder_name}/{self.source_satellite_name}', file) for file in sorted(os.listdir(f'{self.source_root_dir}/{self.source_folder_name}/{self.source_satellite_name}'))]
        self.target_image_paths = [os.path.join(f'{self.target_root_dir}/{self.target_folder_name}/{self.target_satellite_name}', file) for file in sorted(os.listdir(f'{self.target_root_dir}/{self.target_folder_name}/{self.target_satellite_name}'))] if self.target_folder_name is not None else None

    def setup(self, stage=None):
        source_len_paths = len(self.source_image_paths)  
        source_train_length = int(source_len_paths * 0.98)
        source_valid_length = source_len_paths - source_train_length
        
        source_train_paths, source_valid_paths = random_split(self.source_image_paths, [source_train_length, source_valid_length])

        target_train_paths, target_valid_paths = None, None
        if self.target_image_paths is not None:
            target_len_paths = len(self.target_image_paths)  
            target_train_length = int(target_len_paths * 0.98)
            target_valid_length = target_len_paths - target_train_length
            
            target_train_paths, target_valid_paths = random_split(self.target_image_paths, [target_train_length, target_valid_length])

        self.train_dataset = SourceTargetDataset(source_train_paths, target_train_paths, self.transforms['train'], mask_name=self.mask_name, multi_channel=self.multi_channel)
        self.valid_dataset = SourceTargetDataset(source_valid_paths, target_valid_paths, self.transforms['test'], mask_name=self.mask_name, multi_channel=self.multi_channel, load_target_mask=self.load_target_mask)
        self.test_dataset = SourceTargetDataset(self.source_image_paths, self.source_image_paths, self.transforms['test'], is_test=True, mask_name=self.mask_name, load_second_mask_as_target=self.load_second_mask_as_target, multi_channel=self.multi_channel)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.trn_batch_sz, shuffle = True , num_workers = self.num_workers, pin_memory = True)

    def val_dataloader  (self):
        return DataLoader(self.valid_dataset, batch_size = self.tst_batch_sz, shuffle = False, num_workers = self.num_workers, pin_memory = True)

    def test_dataloader (self):
        return DataLoader(self.test_dataset , batch_size = self.tst_batch_sz, shuffle = False, num_workers = self.num_workers, pin_memory = True)
