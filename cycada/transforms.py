import torch
import torchvision.transforms as T
from skimage import transform as tfm
import numpy as np
from kornia.augmentation import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter as CJ, RandomInvert
from kornia.geometry.transform import resize

class Resize(object):

    def __init__(self, image_size: (int, tuple) = 256):

        """
        Parameters:
            image_size: Final size of the image
        """

        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")


    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        source_image, target_image, source_mask, target_mask = sample['source_image'], sample['target_image'], sample['source_mask'], sample['target_mask']

        source_image = resize(source_image, self.image_size)
        source_image = np.clip(source_image, a_min = 0., a_max = 1.)

        if source_mask is not None:
            source_mask = resize(source_mask, self.image_size)

        if target_mask is not None:
            target_mask = resize(target_mask, self.image_size)

        if target_image is not None:
            target_image = resize(target_image, self.image_size)
            target_image = np.clip(target_image, a_min = 0., a_max = 1.)

        return { 'source_image': source_image, 'target_image': target_image, 'source_mask': source_mask, 'target_mask': target_mask }

class ColorJitter(object):
    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        source_image, target_image, source_mask, target_mask = sample['source_image'], sample['target_image'], sample['source_mask'], sample['target_mask']

        invert = RandomInvert(p=0.1)

        jitter = CJ(.1, .1, .3, .5, p=0.8)
        source_image_masked = torch.zeros_like(source_image)
        source_image_masked[:3] = invert(jitter(source_image[:3])[0])[0]
        source_image_masked[3] = source_image[3] 

        jitter = CJ(0, 0, .3, .5, p=0.8)
        target_image_masked = torch.zeros_like(target_image)
        target_image_masked[:3] = invert(jitter(target_image[:3])[0])[0]
        target_image_masked[3] = target_image[3] 

        return { 'source_image': source_image_masked, 'target_image': target_image_masked, 'source_mask': source_mask, 'target_mask': target_mask }

class Random_Flip(object):

    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        source_image, target_image, source_mask, target_mask = sample['source_image'], sample['target_image'], sample['source_mask'], sample['target_mask']

        h_flip_source = RandomHorizontalFlip(True, keepdim=True)
        v_flip_source = RandomVerticalFlip(True, keepdim=True)

        source_image = h_flip_source(v_flip_source(source_image)[0])[0]
        if source_mask is not None:
            source_mask = h_flip_source(v_flip_source(source_mask, params=v_flip_source._params)[0], params=h_flip_source._params)[0]

        h_flip_target = RandomHorizontalFlip(True, keepdim=True)
        v_flip_target = RandomVerticalFlip(True, keepdim=True)

        if target_image is not None:
            target_image = h_flip_target(v_flip_target(target_image)[0])[0]

        if target_mask is not None:
            target_mask = h_flip_target(v_flip_target(target_mask, params=v_flip_target._params)[0], params=h_flip_target._params)[0]

        return { 'source_image': source_image, 'target_image': target_image, 'source_mask': source_mask, 'target_mask': target_mask }

class To_Tensor(object):

    def __call__(self, sample):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        source_image = np.transpose(sample['source_image'].astype(np.float, copy = True), (2, 0, 1))
        source_image = torch.tensor(source_image, dtype = torch.float)
        
        target_image = sample['target_image']
        if target_image is not None:
            target_image = np.transpose(target_image.astype(np.float, copy = True), (2, 0, 1))
            target_image = torch.tensor(target_image, dtype = torch.float)

        source_mask = sample['source_mask']
        if source_mask is not None:
            source_mask = torch.from_numpy(source_mask).float().permute(2, 0, 1)

        target_mask = sample['target_mask']
        if target_mask is not None:
            target_mask = torch.from_numpy(target_mask).float().permute(2, 0, 1)

        return { 'source_image': source_image, 'target_image': target_image, 'source_mask': source_mask, 'target_mask': target_mask }


class Normalize(object):

    def __init__(self, mean = [0.5] * 4, stdv = [0.5] * 4):

        """
        Parameters:
            mean: Normalizing mean
            stdv: Normalizing stdv
        """

        mean = torch.tensor(mean, dtype = torch.float)
        stdv = torch.tensor(stdv, dtype = torch.float)
        self.transforms = T.Normalize(mean = mean, std = stdv)


    def __call__(self, sample):
        source_image, target_image, source_mask, target_mask = sample['source_image'], sample['target_image'], sample['source_mask'], sample['target_mask']
       
        source_image = self.transforms(source_image)
        target_image = self.transforms(target_image) if target_image is not None else None

        return { 'source_image': source_image, 'target_image': target_image if target_image is not None else [], 'source_mask': source_mask, 'target_mask':  target_mask if target_mask is not None else []}

