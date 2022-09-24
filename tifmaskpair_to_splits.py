import torch
import skimage
import skimage.transform
from skimage import io
import os
from zipfile import ZipFile
from os.path import exists
import numpy as np

def split_images(imgs, kernel_size=300, stride=300):
    if len(imgs.shape) == 3:
        imgs = imgs[None, :, :, :]
    
    B, C, H, W = imgs.shape
    
    patches = imgs.float().unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0, 1, 2, 3, 5, 4)

    return patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2] * patches.shape[3], patches.shape[4], patches.shape[5]).squeeze(0).transpose(1, 0)

def flatten(t):
    return [item for sublist in t for item in sublist]

load_pattern = flatten([[
    f'data{"/*" * i}.png',
    f'data{"/*" * i}.tif',
] for i in range(1, 10)])



def load_func(file_path):
    if 'zip' in file_path:
        if not exists(file_path[:-4]):
            with ZipFile(file_path, 'r') as zipObj:
                zipObj.extractall(('/').join(file_path.split('/')[:-1]))

        file_path = file_path[:-4]

    if 'png' in file_path:
        image = io.imread(file_path, as_gray=True)
        image = image.round()
        image = image[:, :, None]
    else:
        image = io.imread(file_path)

    if not 'Ortho' in file_path:
        #image = image.astype(np.uint8)
        image = skimage.transform.resize(image, (7500, 7500))
    
    image = skimage.util.img_as_ubyte(image)

    return torch.from_numpy(image).permute(2, 0, 1)

image_collection = io.ImageCollection(load_pattern, load_func=load_func)

os.makedirs('splits', exist_ok=True)
os.makedirs('splits/masks/buildings', exist_ok=True)
os.makedirs('splits/masks/streets', exist_ok=True)
os.makedirs('splits/satellite/usa', exist_ok=True)
os.makedirs('splits/satellite/other', exist_ok=True)

for i, image in enumerate(image_collection):
    image_path = image_collection.files[i]
    image_name = image_path.split('/')[-1][:-4]

    if 'buildings' in image_name:
        folder = 'masks/buildings'
        image_name = image_name.replace('_buildings', '')
    elif 'streets' in image_name:
        folder = 'masks/streets'
        image_name = image_name.replace('_streets', '')
    elif 'Ortho' in image_path:
        country = image_path.split("/")[-3]
        image_name = image_name[:-5]

        os.makedirs(f'splits/satellite/other/{country}', exist_ok=True)
        folder = f'satellite/other/{country}'
    else:
        folder = 'satellite/usa'

    image_splits = split_images(image).to(torch.uint8) if not 'Ortho' in image_path else split_images(image, 600, 600).to(torch.uint8)

    for split_index, image_split in enumerate(image_splits):
        image_split = image_split.permute(1, 2, 0)
        image_split = skimage.util.img_as_ubyte(skimage.transform.resize(image_split, (300, 300))) if 'Ortho' in image_path else image_split

        io.imsave(f'./splits/{folder}/{image_name}_{split_index}.png', image_split, check_contrast=False)
