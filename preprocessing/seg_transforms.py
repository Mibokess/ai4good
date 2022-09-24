import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

"""For semantic segmentation mask transforms
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
"""

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, segmentation):
        for t in self.transforms:
            image, segmentation = t(image, segmentation)
        return image, segmentation

class Resize:
    def __init__(self, img_size):
        self.img_size = img_size
        self.resize=T.Resize(img_size)
        self.resize_sem=T.Resize(img_size, interpolation=0)

    def __call__(self, image, segmentation):
        image = self.resize(image)
        segmentation = self.resize_sem(segmentation)
        return image, segmentation

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, segmentation):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            segmentation = F.hflip(segmentation)
        return image, segmentation

class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, segmentation):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            segmentation = F.vflip(segmentation)
        return image, segmentation

class PILToTensor:
    def __call__(self, image, segmentation):
        image = F.pil_to_tensor(image)
        segmentation = torch.as_tensor(np.array(segmentation), dtype=torch.int64)
        return image, segmentation

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, segmentation):
        # this might create some values out of bounds
        image = torch.round(F.normalize(image.float(), mean=self.mean, std=self.std))
        # image=torch.clip(image,min=0,max=255)
        return image, segmentation

class SegImageTransform:
    def __init__(self, img_size=64):
        self.transform = {
            "train": Compose(
                [
                    Resize((img_size, img_size)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    PILToTensor(),
                    Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            "test": Compose(
                [
                    Resize((img_size, img_size)),
                    PILToTensor(),
                    Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            "pseudo": Compose(
                [
                    Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        }

    def __call__(self, img, segmentation, phase="train"):
        img,segmentation = self.transform[phase](img,segmentation)

        return img, segmentation