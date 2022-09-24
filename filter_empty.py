import imageio as iio
import os

scratch_path = '/cluster/scratch/mboss/'
folder_name = 'splits_less'

for image_path in os.listdir(f'{scratch_path}/{folder_name}/masks/streets'):
    streets = iio.imread(f'{scratch_path}/{folder_name}/masks/streets/{image_path}')
    buildings = iio.imread(f'{scratch_path}/{folder_name}/masks/buildings/{image_path}')

    threshold = (300**2) * 0.02
    count = 0
    
    if streets.sum() <= threshold or buildings.sum() <= threshold:
        os.remove(f'{scratch_path}/{folder_name}/masks/streets/{image_path}')
        os.remove(f'{scratch_path}/{folder_name}/masks/buildings/{image_path}')
        os.remove(f'{scratch_path}/{folder_name}/satellite/usa/{image_path}')